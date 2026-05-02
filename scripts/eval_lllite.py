"""Evaluate a trained LLLite checkpoint on a holdout set.

Outputs an HTML report with side-by-side panels (cond | generated | target)
plus per-sample metrics:

  * Edge IoU (Canny)        — overlap of edges between generated and target
  * Mean abs depth diff     — for depth conditioning
  * LPIPS (optional)        — perceptual similarity, if `lpips` is installed

Usage:
    python scripts/eval_lllite.py \
        --model models/DreamLite-mobile \
        --weights runs/canny/lllite_canny_step012000.safetensors \
        --eval_dir data/eval_canny \
        --cond_type canny \
        --out_dir runs/canny/eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.environ.get("DREAMLITE_ROOT", "../DreamLite"))

from dreamlite_lllite import DreamLiteMobileLLLitePipeline  # noqa: E402
from dreamlite_lllite.conditioning import preprocess as preprocess_cond  # noqa: E402


def edge_iou(a: Image.Image, b: Image.Image, low: int = 100, high: int = 200) -> float:
    import cv2
    aa = cv2.Canny(cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY), low, high)
    bb = cv2.Canny(cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY), low, high)
    am = aa > 0
    bm = bb > 0
    inter = (am & bm).sum()
    union = (am | bm).sum()
    return float(inter) / float(max(1, union))


def depth_diff(a: Image.Image, b: Image.Image) -> float:
    """Run depth on both, return mean absolute difference normalised to [0, 1]."""
    from dreamlite_lllite.conditioning import preprocess
    da = preprocess("depth", a, size=512, device="cuda")
    db = preprocess("depth", b, size=512, device="cuda")
    return float(((da - db).abs().mean()).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DreamLite-mobile")
    ap.add_argument("--weights", required=True, help="trained LLLite .safetensors")
    ap.add_argument("--eval_dir", required=True,
                    help="directory of eval images (any layout); each .png/.jpg gets one row")
    ap.add_argument("--cond_type", choices=["canny", "depth", "pose"], default="canny")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=12, help="number of eval samples")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cond_emb_dim", type=int, default=32,
                    help="must match the trained checkpoint (canny=32, depth=16)")
    ap.add_argument("--mlp_dim", type=int, default=64,
                    help="must match the trained checkpoint (canny=64, depth=32)")
    ap.add_argument("--multipliers", default="0.0,0.7,1.0",
                    help="comma-separated multipliers to compare in side-by-side panels")
    ap.add_argument("--seed", type=int, default=12345,
                    help="eval-time seed (offset from training seeds so the noise differs)")
    ap.add_argument("--use_caption_sidecar", action="store_true",
                    help="if image has a .txt sidecar, use it as the prompt")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    multipliers = [float(x) for x in args.multipliers.split(",")]
    print("loading pipeline…")
    pipe = DreamLiteMobileLLLitePipeline.from_pretrained(
        args.model, torch_dtype=dtype, device=args.device, multiplier=multipliers[0],
        cond_emb_dim=args.cond_emb_dim, mlp_dim=args.mlp_dim,
    )
    pipe.load_lllite_weights(args.weights, strict=True)

    eval_files = sorted([
        p for p in Path(args.eval_dir).rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ])[: args.n]
    print(f"running on {len(eval_files)} eval images")

    rows: List[dict] = []
    t0 = time.time()
    for i, target_path in enumerate(eval_files):
        target = Image.open(target_path).convert("RGB").resize(
            (args.size, args.size), Image.Resampling.LANCZOS
        )
        # 1. Derive cond image from target (use target as the source of truth)
        cond_t = preprocess_cond(args.cond_type, target, size=args.size)
        cond_pil = Image.fromarray(
            ((cond_t + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        )

        # 2. Caption: sidecar or generic
        caption = "a photograph"
        if args.use_caption_sidecar:
            sidecar = target_path.with_suffix(".txt")
            if sidecar.exists():
                caption = sidecar.read_text(encoding="utf-8").strip()

        # 3. Generate one image PER multiplier, all from the SAME fresh seed
        #    (different from training seed) so any difference is attributable
        #    to LLLite strength, not noise variation.
        sid = f"{i:03d}"
        target.save(out / f"{sid}_target.png")
        cond_pil.save(out / f"{sid}_cond.png")
        pipe.set_cond_image(cond_t.unsqueeze(0))
        per_mult: dict[str, dict] = {}
        for mult in multipliers:
            pipe.set_multiplier(mult)
            gen = pipe(
                prompt=caption,
                num_inference_steps=args.steps,
                height=args.size, width=args.size,
                generator=torch.Generator("cpu").manual_seed(args.seed + i),
            ).images[0]
            tag = f"m{mult:g}".replace(".", "")
            gen.save(out / f"{sid}_gen_{tag}.png")
            if args.cond_type == "canny":
                metric = edge_iou(target, gen)
                metric_name = "edge_iou"
            elif args.cond_type == "depth":
                metric = depth_diff(target, gen)
                metric_name = "depth_abs_diff"
            else:
                metric = float("nan")
                metric_name = "n/a"
            per_mult[tag] = {"multiplier": mult, "metric": metric, "image": f"{sid}_gen_{tag}.png"}

        rows.append({
            "id": sid,
            "target": target_path.name,
            "caption": caption,
            "metric_name": metric_name,
            "per_multiplier": per_mult,
        })
        # log primary (last) multiplier metric
        primary_metric = per_mult[list(per_mult)[-1]]["metric"]
        print(f"  [{i+1}/{len(eval_files)}] {metric_name}@m={multipliers[-1]:.2f} = {primary_metric:.4f}")

    # Per-multiplier mean metric
    means: dict[str, float] = {}
    for tag in rows[0]["per_multiplier"]:
        vals = [r["per_multiplier"][tag]["metric"] for r in rows]
        vals = [v for v in vals if not (v != v)]
        if vals:
            means[tag] = float(np.mean(vals))
    print("means:", {k: round(v, 4) for k, v in means.items()})

    # Write HTML report (side-by-side multipliers)
    html = [f"<html><body><h1>LLLite eval — {args.cond_type}</h1>"]
    html.append("<p>multipliers: " + ", ".join(
        f"<b>{tag}</b>={means.get(tag, float('nan')):.4f}" for tag in means
    ) + " (mean " + rows[0]["metric_name"] + ")</p>")
    html.append("<table border=1 cellpadding=4>")
    cols = ["id", "cond"] + [f"gen {tag}" for tag in rows[0]["per_multiplier"]] + ["target", "caption"]
    html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
    for r in rows:
        sid = r["id"]
        cells = [sid, f"<img src='{sid}_cond.png' width=240>"]
        for tag, info in r["per_multiplier"].items():
            cells.append(
                f"<img src='{info['image']}' width=240><br>"
                f"<small>m={info['multiplier']:g}, "
                f"{r['metric_name']}={info['metric']:.4f}</small>"
            )
        cells += [f"<img src='{sid}_target.png' width=240>", r["caption"]]
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    html.append("</table></body></html>")
    (out / "report.html").write_text("\n".join(html), encoding="utf-8")

    with (out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out/'report.html'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
