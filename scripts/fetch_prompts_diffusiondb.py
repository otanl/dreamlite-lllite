"""Pull a prompt corpus for synthetic LLLite training data.

Default source is DiffusionDB-2M (poloclub/diffusiondb), which is
streamable, large, and contains a wide stylistic distribution.

Output: prompts.jsonl, one row per prompt:

    {"id": "...", "prompt": "..."}

Filters: dedupe (case-insensitive), length window, NSFW score cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm


def _normalise(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip().lower()
    return p


def _is_garbage(p: str) -> bool:
    if len(p) < 8:
        return True
    if len(p.split()) < 3:
        return True
    # Strip common token soup ("trending on artstation, 8k, ..." chains
    # are fine, but reject lines that are >70% commas/symbols).
    alpha = sum(c.isalpha() for c in p)
    if alpha / max(1, len(p)) < 0.4:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output prompts.jsonl path")
    ap.add_argument("--n", type=int, default=4000, help="number of prompts to keep")
    ap.add_argument("--source", default="Gustavosta/Stable-Diffusion-Prompts",
                    help=("HF dataset id. Default: Gustavosta/Stable-Diffusion-Prompts (~73k prompts, "
                          "no auth, parquet — works with current `datasets`). "
                          "DiffusionDB (poloclub/diffusiondb) used to be the canonical source but "
                          "ships a deprecated loading script that newer `datasets` rejects."))
    ap.add_argument("--config", default=None,
                    help="HF config name; leave empty for default")
    ap.add_argument("--max_chars", type=int, default=300)
    ap.add_argument("--min_chars", type=int, default=12)
    ap.add_argument("--max_nsfw", type=float, default=1e-3,
                    help="DiffusionDB image_nsfw threshold; lower = stricter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("install: pip install datasets", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"streaming {args.source}:{args.config} ...")
    if args.config:
        ds = load_dataset(args.source, args.config, split="train", streaming=True)
    else:
        ds = load_dataset(args.source, split="train", streaming=True)
    seen: set[str] = set()
    rows: list[dict] = []

    for ex in tqdm(ds, desc="filter"):
        prompt = (
            ex.get("prompt")
            or ex.get("Prompt")
            or ex.get("text")
            or ex.get("caption")
            or ""
        )
        if not isinstance(prompt, str):
            continue
        if not (args.min_chars <= len(prompt) <= args.max_chars):
            continue
        if _is_garbage(prompt):
            continue
        # DiffusionDB-specific NSFW gate
        nsfw = ex.get("image_nsfw")
        if nsfw is not None and nsfw > args.max_nsfw:
            continue
        norm = _normalise(prompt)
        if norm in seen:
            continue
        seen.add(norm)
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]
        rows.append({"id": h, "prompt": prompt.strip()})
        if len(rows) >= args.n:
            break

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
