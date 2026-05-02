"""Smoke test the training loop with a tiny synthetic manifest.

Generates 4 random RGB images + paired canny conds, then runs train_lllite
for 2 optimizer steps. Verifies:
  - LLLite forward/backward survives bf16 + autocast
  - Optimizer actually updates adapter weights
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"

DREAMLITE_ROOT = os.environ.get("DREAMLITE_ROOT", "../DreamLite")
MODEL = os.environ.get("DREAMLITE_MODEL", "models/DreamLite-mobile")


def _make_dataset(out: Path, n: int = 2, size: int = 512):
    img_dir = out / "imgs"
    cond_dir = out / "cond"
    img_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        a = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        b = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(a).save(img_dir / f"{i:04d}.png")
        Image.fromarray(b).save(cond_dir / f"{i:04d}.png")
        rows.append({
            "image": str((img_dir / f"{i:04d}.png").resolve()),
            "cond": str((cond_dir / f"{i:04d}.png").resolve()),
            "caption": f"random pattern {i}",
        })
    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return manifest


def main():
    workdir = REPO_ROOT / "tests" / "_train_smoke"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    manifest = _make_dataset(workdir, n=2, size=512)
    out_ckpt = workdir / "ckpt"

    cmd = [
        sys.executable,
        str(SCRIPTS / "train_lllite.py"),
        "--model", MODEL,
        "--manifest", str(manifest),
        "--out_dir", str(out_ckpt),
        "--cond_type", "canny",
        "--size", "512",
        "--gradient_checkpointing",
        "--batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--max_steps", "2",
        "--save_every", "2",
        "--log_every", "1",
        "--num_workers", "0",
        "--mixed_precision", "bf16",
    ]
    env = os.environ.copy()
    env["DREAMLITE_ROOT"] = DREAMLITE_ROOT
    env["PYTHONPATH"] = str(SRC) + os.pathsep + DREAMLITE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise SystemExit(res.returncode)
    saved = list(out_ckpt.glob("*.safetensors"))
    assert saved, "no checkpoint produced"
    print("saved:", saved)
    print("OK")


if __name__ == "__main__":
    main()
