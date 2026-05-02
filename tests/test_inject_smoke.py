"""Smoke test: load DreamLite-mobile, attach LLLite, do a forward pass.

Not pytest — run directly. Uses zero-init adapter (default), so the output
must equal the vanilla pipeline's output for the same seed.
"""

import os
import sys
import argparse

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
# DreamLite is not pip-installable; add the cloned source dir to path.
DREAMLITE_ROOT = os.environ.get("DREAMLITE_ROOT", "../DreamLite")
sys.path.insert(0, DREAMLITE_ROOT)

from dreamlite_lllite import (  # noqa: E402
    apply_lllite,
    list_lllite_targets,
    DreamLiteMobileLLLitePipeline,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/DreamLite-mobile")
    parser.add_argument("--prompt", default="a photo of a corgi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print("=== Step 1: enumerate targets ===")
    # Quick enumerate without loading conditioning encoders (just structural)
    from dreamlite import DreamLiteMobilePipeline
    base = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)

    targets = list_lllite_targets(base.unet)
    print(f"found {len(targets)} target Linear layers")
    by_attn = {"attn1.to_q": 0, "attn1.to_k": 0, "attn1.to_v": 0, "attn2.to_q": 0}
    for t in targets:
        key = f"{t.attn_kind}.{t.proj_kind}"
        by_attn[key] = by_attn.get(key, 0) + 1
    print(" by kind:", by_attn)
    by_block = {}
    for t in targets:
        by_block[t.block_root] = by_block.get(t.block_root, 0) + 1
    print(" by block:", by_block)

    print("=== Step 2: probe & inject ===")
    controller = apply_lllite(
        base.unet,
        cond_emb_dim=32,
        mlp_dim=64,
        cond_image_size=1024,
    )
    controller.to(device=args.device, dtype=dtype)
    print(f"attached {len(controller)} LLLite modules")
    # Check feature_hw distribution
    print(" feature sizes per block:")
    seen = {}
    for name, m in controller.modules_dict.items():
        seen.setdefault(m.block_feature_size, 0)
        seen[m.block_feature_size] += 1
    print(" ", seen)

    n_params = controller.num_parameters()
    print(f"trainable params (zero-init): {n_params:,}  (~{n_params/1e6:.2f} M)")

    print("=== Step 3: generate WITHOUT cond image (controller idle) ===")
    out_a = base(
        prompt=args.prompt,
        num_inference_steps=4,
        height=1024, width=1024,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    out_a.save("smoke_no_cond.png")

    print("=== Step 4: generate WITH cond image (zero-init -> identical output) ===")
    cond = torch.zeros(1, 3, 1024, 1024, device=args.device, dtype=dtype)
    controller.set_cond_image(cond)
    out_b = base(
        prompt=args.prompt,
        num_inference_steps=4,
        height=1024, width=1024,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    out_b.save("smoke_zero_cond.png")

    # Pixel-equality check (allowing small fp tolerance)
    import numpy as np
    a = np.asarray(out_a, dtype=np.int16)
    b = np.asarray(out_b, dtype=np.int16)
    diff = np.abs(a - b).mean()
    print(f"mean abs pixel diff with zero-init adapter: {diff:.4f}  (should be ~0)")
    assert diff < 1.0, "zero-init LLLite should not perturb output"

    print("=== Step 5: detach LLLite, confirm UNet returns to vanilla ===")
    controller.remove_from()
    out_c = base(
        prompt=args.prompt,
        num_inference_steps=4,
        height=1024, width=1024,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    out_c.save("smoke_detached.png")
    c = np.asarray(out_c, dtype=np.int16)
    diff_ac = np.abs(a - c).mean()
    print(f"mean abs pixel diff after detach vs original: {diff_ac:.4f}  (should be 0)")
    assert diff_ac < 1.0, "after detach UNet should be byte-identical"

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
