"""Compute metrics for a cached run (reads only outputs/<method>/<dataset>/ and the GT masks).

Usage:
    python eval/evaluate.py --run outputs/anomalyclip/mvtec --dataset mvtec --paths kaggle

Writes metrics.json: per-category metrics and their mean (the dataset number, as in the papers).
Pixel metrics are computed at EVAL_SIZE x EVAL_SIZE for every method (override: --eval-size).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data.registry import build_dataset  # noqa: E402
from eval.cache import EVAL_SIZE, load_map, match_records, read_scores  # noqa: E402
from eval.metrics import category_metrics  # noqa: E402

METRICS = ("image_auroc", "image_f1_max", "pixel_auroc", "aupro")


def evaluate_run(
    rdir: Path,
    dataset_name: str,
    paths: str = "local",
    dataset_overrides: list[str] | None = None,
    limit: int | None = None,
    eval_size: int = EVAL_SIZE,
) -> dict:
    rdir = Path(rdir)
    dataset = build_dataset(dataset_name, paths, dataset_overrides)
    rows = read_scores(rdir)
    index_of = match_records(dataset, rdir, rows)

    per_category = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        labels = np.array([r["label"] for r in cat_rows])
        scores = np.array([r["score"] for r in cat_rows])
        if len(set(labels)) < 2:
            print(f"  {cat}: skipped (needs both normal and anomalous images)")
            continue
        masks = maps = None
        if dataset.has_pixel_labels:
            masks = np.stack([dataset.load_mask(index_of[r["path"]], eval_size) for r in cat_rows])
            maps = np.stack([load_map(rdir, r["path"]) for r in cat_rows])
        m = category_metrics(labels, scores, masks, maps)
        m["n_images"], m["n_anomalous"] = len(cat_rows), int(labels.sum())
        per_category[cat] = m

    mean = {k: float(np.mean([m[k] for m in per_category.values()])) if per_category else math.nan for k in METRICS}
    result = {
        "dataset": dataset_name,
        "n_images": len(rows),
        "n_categories": len(per_category),
        "eval_size": eval_size,
        "complete": len(rows) == len(dataset) and limit is None,
        "mean": mean,
        "per_category": per_category,
    }
    (rdir / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\n{rdir}  ({len(rows)} images{'' if result['complete'] else ', INCOMPLETE run'})")
    print(f"{'category':22s}" + "".join(f"{k:>14s}" for k in METRICS))
    for cat, m in per_category.items():
        print(f"{cat:22s}" + "".join(f"{100 * m[k]:14.1f}" for k in METRICS))
    print(f"{'MEAN':22s}" + "".join(f"{100 * mean[k]:14.1f}" for k in METRICS))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--paths", default="local")
    p.add_argument("--eval-size", type=int, default=EVAL_SIZE, help="must match the size the maps were saved at")
    a = p.parse_args()
    evaluate_run(Path(a.run), a.dataset, a.paths, eval_size=a.eval_size)


if __name__ == "__main__":
    main()
