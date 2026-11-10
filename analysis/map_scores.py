"""Image scores derived from each method's own anomaly map (Phase 5, RQ4 at image level).

Every method outputs both an image score and a map. For each image we also score it from the map:
the map's maximum, and the mean of its top 1% and top 5% pixels (VisualAD's image score is the top-k
mean of its map). If the map-derived score separates normal from anomalous images better than the
method's own image score, the map "knows" something the image score ignores.

Usage (Kaggle CPU job, run outputs attached):
    python analysis/map_scores.py --out /kaggle/working/map_scores
Writes <out>/<method>/<dataset>.csv with: image (path relative to the dataset root), category, label,
score (the method's own), map_max, map_top1, map_top5.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from eval.cache import load_map, read_scores, relative_path, run_dataset_root  # noqa: E402
from methods.base import topk_mean  # noqa: E402


def map_scores(amap: np.ndarray) -> dict[str, float]:
    return {"map_max": float(amap.max()), "map_top1": topk_mean(amap, 0.01), "map_top5": topk_mean(amap, 0.05)}


def main() -> None:
    from bench_stats import find_run

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/map_scores")
    a = p.parse_args()

    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    for method, by_dataset in manifest.runs.items():
        for dataset, src in by_dataset.items():
            rdir = find_run(Path(a.input), src.kernel, src.run)
            print(f"===== {method} on {dataset}: {rdir}", flush=True)
            root = run_dataset_root(rdir)
            rows = [
                {
                    "image": relative_path(r["path"], root),
                    "category": r["category"],
                    "label": r["label"],
                    "score": r["score"],
                    **map_scores(load_map(rdir, r["path"])),
                }
                for r in read_scores(rdir)
            ]
            out = Path(a.out) / method / f"{dataset}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("FAILED: none")


if __name__ == "__main__":
    main()
