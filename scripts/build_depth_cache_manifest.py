"""Build a cache manifest for depth training by reusing canny's latent shards.

Latents and text embeds are independent of cond_type, so the .pt shards
produced for canny are reusable. We only need to swap cond paths.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("usage: build_depth_cache_manifest.py <canny_cache_manifest.jsonl> <depth_cache_manifest.jsonl>")
        sys.exit(2)
    src, dst = sys.argv[1:3]
    n = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cond = row.get("cond", "")
            # Swap any "/canny/" or "\canny\" -> "/depth/" or "\depth\"
            cond = cond.replace("/canny/", "/depth/").replace("\\canny\\", "\\depth\\")
            row["cond"] = cond
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {dst}")


if __name__ == "__main__":
    main()
