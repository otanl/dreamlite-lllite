"""Generate synthetic LLLite training images with DreamLite-mobile.

This is the recommended LLLite workflow per kohya-ss: feed prompts into the
*base* model, save the generated images alongside their prompts as captions,
then derive conditioning images (canny/depth/pose) from those generated
images. The training set thus matches the model's own style distribution.

Usage:
    python scripts/generate_training_images.py \
        --model models/DreamLite-mobile \
        --prompts data/prompts.jsonl \
        --out data/canny/imgs \
        --size 1024 --steps 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.environ.get("DREAMLITE_ROOT", "../DreamLite"))


def _load_prompts(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl" or line.startswith("{"):
                rows.append(json.loads(line))
            else:
                # plain text: one prompt per line
                rows.append({"prompt": line})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DreamLite-mobile")
    ap.add_argument("--prompts", required=True, help=".jsonl or .txt prompt list")
    ap.add_argument("--out", required=True, help="output dir for images + .txt sidecars")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4, help="DreamLite-mobile is 4-step")
    ap.add_argument("--seed_base", type=int, default=0,
                    help="per-prompt seed = seed_base + index, for reproducibility")
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip prompts whose output PNG already exists")
    ap.add_argument("--quantize_te", action="store_true",
                    help="load text encoder in NF4 (saves VRAM, slower load)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = _load_prompts(Path(args.prompts))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} prompts queued; output -> {out}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print("loading DreamLite-mobile…")
    t0 = time.time()
    if args.quantize_te:
        # Optional 4-bit text encoder load (matches our measure_vram path).
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, AutoTokenizer
        try:
            from transformers import Qwen3VLProcessor
        except Exception:
            Qwen3VLProcessor = None  # type: ignore[assignment]
        from diffusers.models import AutoencoderTiny
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from dreamlite.models import DreamLiteUNetModel
        from dreamlite import DreamLiteMobilePipeline

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(args.model, "text_encoder"),
            quantization_config=bnb, torch_dtype=dtype,
        )
        tok = AutoTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))
        proc = Qwen3VLProcessor.from_pretrained(os.path.join(args.model, "processor")) if Qwen3VLProcessor else None
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.model, "scheduler"))
        vae = AutoencoderTiny.from_pretrained(os.path.join(args.model, "vae"), torch_dtype=dtype)
        unet = DreamLiteUNetModel.from_pretrained(os.path.join(args.model, "unet"), torch_dtype=dtype)
        pipe = DreamLiteMobilePipeline(
            text_encoder=te, tokenizer=tok, processor=proc,
            scheduler=sched, vae=vae, unet=unet,
        )
        pipe.vae.to(args.device)
        pipe.unet.to(args.device)
    else:
        from dreamlite import DreamLiteMobilePipeline
        pipe = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    n_done = 0
    n_skip = 0
    n_err = 0
    for i, row in enumerate(tqdm(rows, desc="generate")):
        prompt = row["prompt"]
        rid = row.get("id") or f"{i:06d}"
        img_path = out / f"{rid}.png"
        cap_path = out / f"{rid}.txt"

        if args.skip_existing and img_path.exists():
            n_skip += 1
            continue
        try:
            seed = args.seed_base + i
            img = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                height=args.size,
                width=args.size,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            img.save(img_path)
            cap_path.write_text(prompt, encoding="utf-8")
            n_done += 1
        except Exception as e:
            print(f"  ERR {rid}: {e}")
            n_err += 1

    print(f"done: generated={n_done} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
