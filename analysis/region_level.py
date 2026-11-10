"""Region- and image-level failure summaries from the Kaggle tables of analysis/regions.py (RQ1, RQ4).

Inputs: outputs/kaggle/zsad-p5-regions/analysis/  (geometry/<dataset>.csv, <method>/<dataset>_{regions,images}.csv)
        and the per-image scores of every run (for the 5% image-FPR operating point).
Outputs: analysis/results/region_found_by_size.csv, region_found_by_contrast.csv, peak_in_mask.csv

Definitions (see analysis/regions.py for how the columns were computed)
- A region is *found* if at least half its pixels score above the method's 1%-FPR pixel threshold.
- Size bins (pixels at 256 x 256): tiny < 100, small 100-1,000, medium 1,000-5,000, large > 5,000.
- Contrast bins (|mean grey inside - mean grey of a 5-px ring|, 0-255): low < 10, medium 10-30, high > 30.
- Right image, wrong place (RQ4): an anomalous image the method flags at 5% image FPR, but whose map
  peak is more than 4 px from every defect pixel.
CIs: 95% percentile bootstrap, 1,000 resamples of *images* (regions of one image stay together).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.image_level import fpr_threshold, load_runs  # noqa: E402
from eval.bootstrap import percentile_ci  # noqa: E402

TABLES = REPO / "outputs" / "kaggle" / "zsad-p5-regions" / "analysis"
OUT = REPO / "analysis" / "results"
N_RESAMPLES = 1000
FOUND_OVERLAP = 0.5
SIZE_BINS = [
    (0, 100, "tiny (<100 px)"),
    (100, 1000, "small (100-1k)"),
    (1000, 5000, "medium (1k-5k)"),
    (5000, 10**9, "large (>5k)"),
]
CONTRAST_BINS = [(0, 10, "low (<10)"), (10, 30, "medium (10-30)"), (30, 1e9, "high (>30)")]
PIXEL_DATASETS = ["mvtec", "visa", "mpdd", "bmad_brain"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cluster_bootstrap_mean(values: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `values` with a CI from resampling whole groups (images) with replacement."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(float))
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(N_RESAMPLES):
        pick = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        reps.append((pick @ sums) / (pick @ counts))
    lo, hi = percentile_ci(np.array(reps))
    return float(values.mean()), lo, hi


def bin_of(x: float, bins) -> str | None:
    for lo, hi, name in bins:
        if lo <= x < hi:
            return name
    return None


def found_by(methods: list[str], key: str, bins) -> list[dict]:
    out = []
    for d in PIXEL_DATASETS:
        geo = read_csv(TABLES / "geometry" / f"{d}.csv")
        for m in methods:
            reg = read_csv(TABLES / m / f"{d}_regions.csv")
            assert [(g["image"], g["region"]) for g in geo] == [(r["image"], r["region"]) for r in reg], (m, d)
            found = np.array([float(r["overlap_fpr1"]) >= FOUND_OVERLAP for r in reg])
            imgs = np.array([g["image"] for g in geo])
            which = np.array([bin_of(float(g[key]), bins) for g in geo], dtype=object)
            for _, _, name in bins:
                sel = which == name
                mean, lo, hi = cluster_bootstrap_mean(found[sel], imgs[sel])
                out.append(
                    {
                        "dataset": d,
                        "method": m,
                        "bin": name,
                        "n_regions": int(sel.sum()),
                        "found": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return out


def peak_table(methods: list[str], runs: dict) -> list[dict]:
    out = []
    for d in PIXEL_DATASETS:
        for m in methods:
            rows = runs[(m, d)]
            thr = {}
            for c in {r["category"] for r in rows}:
                normal = [r["score"] for r in rows if r["category"] == c and r["label"] == 0]
                thr[c] = fpr_threshold(np.array(normal), 0.05)
            imgs = read_csv(TABLES / m / f"{d}_images.csv")
            flagged = np.array([float(r["score"]) > thr[r["category"]] for r in imgs])
            wrong = np.array([r["peak_inside_tol"] == "False" for r in imgs])
            names = np.array([r["image"] for r in imgs])
            mean, lo, hi = cluster_bootstrap_mean(wrong[flagged], names[flagged])
            all_mean, all_lo, all_hi = cluster_bootstrap_mean(wrong, names)
            out.append(
                {
                    "dataset": d,
                    "method": m,
                    "n_flagged": int(flagged.sum()),
                    "wrong_place_among_flagged": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "peak_off_defect_all_anomalous": all_mean,
                    "all_ci_low": all_lo,
                    "all_ci_high": all_hi,
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def main() -> None:
    runs, manifest = load_runs(REPO / "outputs" / "kaggle")
    methods = [m for m in manifest.methods if not manifest.methods[m].get("transductive", False)]
    write_csv(OUT / "region_found_by_size.csv", found_by(methods, "area", SIZE_BINS))
    write_csv(OUT / "region_found_by_contrast.csv", found_by(methods, "contrast", CONTRAST_BINS))
    write_csv(OUT / "peak_in_mask.csv", peak_table(methods, runs))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
