"""Batched-generation variant of generate_training_images.py.

DreamLite's stock pipeline forces `batch_size=1` (see
pipeline_dreamlite_mobile.py:281). We bypass `__call__` and drive the
components directly with proper batching, which amortises Python/launch
overhead and yields ~2-3x throughput on a 3090-class GPU at modest
batch sizes.

Usage mirrors the single-image script, plus `--batch_size`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.environ.get("DREAMLITE_ROOT", "../DreamLite"))


def _load_prompts(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl" or line.startswith("{"):
                rows.append(json.loads(line))
            else:
                rows.append({"prompt": line})
    return rows


@torch.no_grad()
def generate_batch(pipe, prompts: List[str], height: int, width: int,
                   steps: int, seeds: List[int], dtype: torch.dtype, device: str):
    """Run DreamLite-mobile inference for a list of prompts simultaneously.

    Returns a list of PIL.Image of length len(prompts).
    """
    from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput  # noqa: F401
    from dreamlite.pipelines.dreamlite.pipeline_dreamlite_mobile import (
        retrieve_timesteps, calculate_shift,
    )

    B = len(prompts)
    # 1. Encode all prompts together
    prompt_strs = [f"[Generate]: {p}" for p in prompts]
    embeds, attn_mask = pipe.encode_prompt(
        mode="generate",
        prompts=prompt_strs,
        device=device,
        dtype=dtype,
    )

    # 2. time_ids
    add_time_ids = torch.tensor([[width, height]] * B, device=device, dtype=dtype)

    # 3. Latents (per-sample seed). DreamLite's prepare_latents supports a
    #    single Generator; we replicate per-sample with explicit randn_tensor.
    H = height // pipe.vae_scale_factor
    W = width // pipe.vae_scale_factor
    nch = pipe.vae.config.latent_channels
    gens = [torch.Generator("cpu").manual_seed(s) for s in seeds]
    latents = torch.stack(
        [torch.randn(nch, H, W, generator=g) for g in gens], 0
    ).to(device=device, dtype=dtype)
    image_latents = torch.zeros_like(latents)

    # 4. Scheduler timesteps (FlowMatchEulerDiscreteScheduler)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    image_seq_len = latents.shape[2] * latents.shape[3] // 4
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.16),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, steps, device, sigmas=sigmas, mu=mu,
    )

    # 5. Denoising loop
    for t in timesteps:
        model_input = torch.cat([latents, image_latents], dim=3)
        noise_pred = pipe.unet(
            model_input,
            timestep=t.expand(model_input.shape[0]).to(dtype),
            encoder_hidden_states=embeds,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs={"time_ids": add_time_ids},
            return_dict=False,
        )[0]
        noise_pred = noise_pred[..., : latents.shape[-1]]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # 6. Decode with VAE
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents / pipe.vae.config.scaling_factor) + shift_factor
    images = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype), return_dict=False)[0]

    # 7. Convert to PIL
    images = (images.float() + 1.0) * 0.5
    images = images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 255).astype("uint8")
    return [Image.fromarray(im) for im in images]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DreamLite-mobile")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--quantize_te", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="prompts per UNet forward pass; >1 amortises Python overhead")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = _load_prompts(Path(args.prompts))
    if args.limit:
        rows = rows[: args.limit]
    if args.skip_existing:
        n_before = len(rows)
        rows = [r for i, r in enumerate(rows)
                if not (out / f"{r.get('id') or f'{i:06d}'}.png").exists()]
        print(f"{n_before - len(rows)} already exist, {len(rows)} remaining")
    print(f"queue: {len(rows)} prompts, batch_size={args.batch_size}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print("loading DreamLite-mobile…")
    t0 = time.time()
    if args.quantize_te:
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
            os.path.join(args.model, "text_encoder"), quantization_config=bnb, torch_dtype=dtype,
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
    n_err = 0
    pbar = tqdm(total=len(rows), desc=f"generate@bs={args.batch_size}")
    for chunk_start in range(0, len(rows), args.batch_size):
        chunk = rows[chunk_start: chunk_start + args.batch_size]
        prompts = [r["prompt"] for r in chunk]
        ids = [r.get("id") or f"{chunk_start + j:06d}" for j, r in enumerate(chunk)]
        seeds = [args.seed_base + chunk_start + j for j in range(len(chunk))]
        try:
            imgs = generate_batch(
                pipe, prompts, args.size, args.size, args.steps, seeds, dtype, args.device,
            )
            for rid, prompt, img in zip(ids, prompts, imgs):
                img.save(out / f"{rid}.png")
                (out / f"{rid}.txt").write_text(prompt, encoding="utf-8")
                n_done += 1
                pbar.update(1)
        except Exception as e:
            print(f"  ERR chunk@{chunk_start}: {type(e).__name__}: {e}")
            n_err += len(chunk)
            pbar.update(len(chunk))
    pbar.close()
    print(f"done: generated={n_done} errors={n_err}")


if __name__ == "__main__":
    main()
