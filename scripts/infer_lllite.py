"""Single-image LLLite inference for DreamLite-mobile.

Usage:
    python scripts/infer_lllite.py \
        --model models/DreamLite-mobile \
        --weights checkpoints/canny_step20000.safetensors \
        --cond_type canny \
        --cond_image input.jpg \
        --prompt "a futuristic city" \
        --out output.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image

# Ensure local package is importable when run from a fresh checkout
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.environ.get("DREAMLITE_ROOT", "../DreamLite"))

from dreamlite_lllite import DreamLiteMobileLLLitePipeline
from dreamlite_lllite.conditioning import preprocess


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/DreamLite-mobile",
                   help="Path to DreamLite-mobile diffusers folder")
    p.add_argument("--weights", required=False, default=None,
                   help="Path to LLLite weights .safetensors. If omitted, runs with zero-init adapter (vanilla output).")
    p.add_argument("--cond_type", choices=["canny", "depth", "pose"], default="canny")
    p.add_argument("--cond_image", required=False, default=None,
                   help="Source image to derive the conditioning from")
    p.add_argument("--prompt", default="a photo of a corgi")
    p.add_argument("--out", default="lllite_out.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--cond_emb_dim", type=int, default=32)
    p.add_argument("--mlp_dim", type=int, default=64)
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[1/3] Loading DreamLite-mobile + LLLite controller from {args.model}")
    t0 = time.time()
    pipe = DreamLiteMobileLLLitePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device=args.device,
        cond_emb_dim=args.cond_emb_dim,
        mlp_dim=args.mlp_dim,
        cond_image_size=args.size,
        multiplier=args.multiplier,
    )
    print(f"  loaded in {time.time()-t0:.1f}s, {len(pipe.controller)} adapter modules")

    if args.weights:
        print(f"[2/3] Loading LLLite weights from {args.weights}")
        pipe.load_lllite_weights(args.weights, strict=True)
    else:
        print("[2/3] No --weights given; running with zero-init adapter (vanilla output)")

    cond_tensor = None
    if args.cond_image is not None:
        print(f"  preprocessing cond image with {args.cond_type}")
        cond = preprocess(args.cond_type, args.cond_image, size=args.size)
        cond_tensor = cond.unsqueeze(0)  # (1, 3, H, W)
        # save cond preview alongside output
        prev_path = os.path.splitext(args.out)[0] + f".cond_{args.cond_type}.png"
        cond_img = ((cond + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(cond_img).save(prev_path)
        print(f"  saved cond preview -> {prev_path}")
        pipe.set_cond_image(cond_tensor)
    else:
        print("  no --cond_image; LLLite is loaded but inactive")

    print(f"[3/3] Generating: {args.prompt!r}")
    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    print(f"  generated in {time.time()-t0:.2f}s")
    out.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
