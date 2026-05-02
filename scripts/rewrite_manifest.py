"""Rewrite absolute paths in a manifest.jsonl after moving the data tree.

Usage:
    python rewrite_manifest.py <input.jsonl> <output.jsonl> <old_root> <new_root>
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 5:
        print("usage: rewrite_manifest.py <input> <output> <old_root> <new_root>", file=sys.stderr)
        sys.exit(2)
    src, dst, old, new = sys.argv[1:5]

    # Normalise both forms (forward / backward slash) for safety.
    old_variants = [old.replace("\\", "/"), old.replace("/", "\\"), old]
    new = new.replace("/", "\\")

    n = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in ("image", "cond", "shard"):
                if k in row and isinstance(row[k], str):
                    v = row[k]
                    for o in old_variants:
                        v = v.replace(o, new)
                    row[k] = v
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"rewrote {n} rows -> {dst}")


if __name__ == "__main__":
    main()
