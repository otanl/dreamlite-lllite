"""Pre-encode (image, caption) pairs into cached latent + text embedding tensors.

Speeds up training by ~3-5x: VAE.encode() and Qwen3-VL forward dominate
the per-step cost otherwise. We run them once over the dataset, then the
training loop just reads the cached `.pt` files.

Output layout:

    cache/
    ├── manifest.jsonl     # one row per sample, with paths to all caches
    └── shards/<id>.pt     # dict with keys: latent, prompt_embeds, prompt_mask

Caching is keyed by content hash of (image_path, prompt) so re-runs skip
existing shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.environ.get("DREAMLITE_ROOT", "../DreamLite"))


def _hash_id(image_path: str, prompt: str) -> str:
    h = hashlib.sha1()
    h.update(image_path.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DreamLite-mobile")
    ap.add_argument("--manifest", required=True, help="manifest.jsonl from prepare_dataset.py")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quantize_te", action="store_true",
                    help="load text encoder in NF4 (saves VRAM)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = cache_dir / "manifest.jsonl"

    rows: List[dict] = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"manifest entries: {len(rows)}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # Load DreamLite to access vae + text_encoder + helpers (auto base/mobile)
    from dreamlite_lllite.pipeline import _detect_pipeline_class
    PipelineCls, variant = _detect_pipeline_class(args.model)
    print(f"loading {variant} from {args.model}…")
    if args.quantize_te:
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, AutoTokenizer
        try:
            from transformers import Qwen3VLProcessor
        except Exception:
            Qwen3VLProcessor = None  # type: ignore[assignment]
        from diffusers.models import AutoencoderTiny
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from dreamlite.models import DreamLiteUNetModel

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(args.model, "text_encoder"), quantization_config=bnb, torch_dtype=dtype,
        )
        tok = AutoTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))
        proc = Qwen3VLProcessor.from_pretrained(os.path.join(args.model, "processor")) if Qwen3VLProcessor else None
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.model, "scheduler"))
        vae = AutoencoderTiny.from_pretrained(os.path.join(args.model, "vae"), torch_dtype=dtype)
        unet = DreamLiteUNetModel.from_pretrained(os.path.join(args.model, "unet"), torch_dtype=dtype)
        pipe = PipelineCls(
            text_encoder=te, tokenizer=tok, processor=proc,
            scheduler=sched, vae=vae, unet=unet,
        )
        pipe.vae.to(args.device)
    else:
        pipe = PipelineCls.from_pretrained(args.model, torch_dtype=dtype).to(args.device)

    pipe.vae.eval()
    pipe.text_encoder.eval()
    for p in pipe.vae.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)

    vae_scaling = pipe.vae.config.scaling_factor
    vae_shift = getattr(pipe.vae.config, "shift_factor", 0.0)

    tfm = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    cached_rows: List[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for row in tqdm(rows, desc="cache"):
            img_path = row["image"]
            cond_path = row.get("cond")
            caption = row.get("caption", "")

            cid = _hash_id(img_path, caption)
            shard = shards_dir / f"{cid}.pt"
            if shard.exists():
                cached_rows.append({**row, "shard": str(shard.resolve())})
                continue

            img = Image.open(img_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device=args.device, dtype=dtype)
            latent = pipe.vae.encode(x).latents
            latent = (latent - vae_shift) * vae_scaling

            embeds, mask = pipe.encode_prompt(
                mode="generate",
                prompts=[f"[Generate]: {caption}"],
                device=args.device,
                dtype=dtype,
            )

            torch.save(
                {
                    "latent": latent.squeeze(0).cpu(),
                    "prompt_embeds": embeds.squeeze(0).cpu(),
                    "prompt_mask": mask.squeeze(0).cpu(),
                    "caption": caption,
                    "image": img_path,
                    "cond": cond_path,
                },
                shard,
            )
            cached_rows.append({**row, "shard": str(shard.resolve())})

    with out_manifest.open("w", encoding="utf-8") as f:
        for r in cached_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(cached_rows)} shards to {shards_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
