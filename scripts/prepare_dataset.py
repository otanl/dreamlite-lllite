"""Generate (image, conditioning, caption) triples for LLLite training.

Walks an image directory, derives a conditioning image (canny / depth / pose),
and writes a manifest JSONL. Captions can come from a sidecar .txt file
(filename.txt), a CSV, or a single shared caption.

Output layout:

    out_dir/
    ├── manifest.jsonl        # one row per sample
    └── cond/<rel_path>.png   # conditioning image (3-channel uint8)

Manifest row:

    {"image": "abs/path.jpg", "cond": "abs/path.png", "caption": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from dreamlite_lllite.conditioning import preprocess  # noqa: E402

import torch


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def caption_for(img_path: Path, default: str) -> str:
    side = img_path.with_suffix(".txt")
    if side.exists():
        return side.read_text(encoding="utf-8").strip()
    return default


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Walk a directory of images (with .txt sidecar captions, kohya-ss layout), "
            "derive conditioning images of the chosen type, and write a manifest."
        )
    )
    ap.add_argument("--src", required=True, help="root dir of source images (kohya-ss image_dir)")
    ap.add_argument("--out", required=True, help="output dir; conditioning images go in <out>/cond")
    ap.add_argument("--cond_type", choices=["canny", "depth", "pose"], required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--default_caption", default="",
                    help="used when no .txt sidecar is found")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip images whose conditioning file already exists")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    cond_dir = out / "cond"
    cond_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    rows = []
    paths = list(iter_images(src))
    if args.limit:
        paths = paths[: args.limit]

    for img_path in tqdm(paths, desc=f"preprocess[{args.cond_type}]"):
        try:
            rel = img_path.relative_to(src)
            cond_path = cond_dir / rel.with_suffix(".png")
            cond_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and cond_path.exists():
                cap = caption_for(img_path, args.default_caption)
                rows.append({
                    "image": str(img_path.resolve()),
                    "cond": str(cond_path.resolve()),
                    "caption": cap,
                })
                continue

            cond_t = preprocess(args.cond_type, str(img_path), size=args.size)
            # Save cond as PNG (cond_t in [-1,1])
            cond_uint8 = ((cond_t + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            cond_pil = Image.fromarray(cond_uint8.permute(1, 2, 0).cpu().numpy())
            cond_pil.save(cond_path)

            cap = caption_for(img_path, args.default_caption)
            rows.append({
                "image": str(img_path.resolve()),
                "cond": str(cond_path.resolve()),
                "caption": cap,
            })
        except Exception as e:
            print(f"  skip {img_path}: {e}")

    with manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {manifest}")


if __name__ == "__main__":
    main()
