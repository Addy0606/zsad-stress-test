"""95% bootstrap confidence intervals for the dataset-level metrics (mean over categories).

One bootstrap replicate:
  1. In every category, resample its normal images with replacement (same count) and its
     anomalous images with replacement (same count). Keeping the counts fixed per category and
     label means every replicate has both classes and the same make-up as the real test set.
  2. Compute each metric per category on the resample and average over categories, exactly as
     for the headline number.
Repeat 1,000 times; the CI is the 2.5th-97.5th percentile of the replicates (percentile bootstrap).

Image metrics are recomputed from per-image scores. Pixel metrics use the per-image histograms
from eval/pixel_stats.py, with the same resampled images (each image weighted by how many times
it was drawn), so all four metrics come from the same replicates.
"""

from __future__ import annotations

import numpy as np

from eval.metrics import image_auroc, image_f1_max
from eval.pixel_stats import binned_aupro, binned_pixel_auroc, region_frac_above

METRICS = ("image_auroc", "image_f1_max", "pixel_auroc", "aupro")


def resample_counts(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """How many times each image is drawn in one replicate (stratified by label)."""
    counts = np.zeros(len(labels), dtype=np.int64)
    for value in (0, 1):
        idx = np.flatnonzero(labels == value)
        counts += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(labels))
    return counts


def bootstrap_metrics(
    rows: list[dict],
    pixel_stats: dict[str, dict[str, np.ndarray]] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Replicates of each dataset-level metric: {metric: array of shape (n_resamples,)} (NaN if unavailable).

    rows: scores.csv rows (needs category, label, score, path).
    pixel_stats: output of eval.pixel_stats.load_stats, or None for image-only datasets.
    """
    rng = np.random.default_rng(seed)
    categories = sorted({r["category"] for r in rows})
    per_cat = {m: np.full((n_resamples, len(categories)), np.nan) for m in METRICS}

    for c, cat in enumerate(categories):
        cat_rows = [r for r in rows if r["category"] == cat]
        labels = np.array([r["label"] for r in cat_rows])
        scores = np.array([r["score"] for r in cat_rows])
        if len(set(labels)) < 2:
            continue
        s = (pixel_stats or {}).get(cat)
        if s is not None:
            row_of = {p: i for i, p in enumerate(s["paths"].tolist())}
            perm = np.array([row_of[r["path"]] for r in cat_rows])  # scores-row i -> stats row perm[i]
            frac_above = region_frac_above(s["reg"])
        for b in range(n_resamples):
            counts = resample_counts(labels, rng)
            idx = np.repeat(np.arange(len(labels)), counts)
            per_cat["image_auroc"][b, c] = image_auroc(labels[idx], scores[idx])
            per_cat["image_f1_max"][b, c] = image_f1_max(labels[idx], scores[idx])
            if s is not None:
                w = np.zeros(len(perm))
                w[perm] = counts  # weight of each stats row
                neg, pos = w @ s["neg"], w @ s["pos"]
                per_cat["pixel_auroc"][b, c] = binned_pixel_auroc(neg, pos)
                per_cat["aupro"][b, c] = binned_aupro(neg, s["reg"], w[s["reg_img"]], frac_above=frac_above)

    out = {}
    for m, arr in per_cat.items():
        valid = ~np.isnan(arr).all(axis=0)  # categories where the metric exists
        out[m] = arr[:, valid].mean(axis=1) if valid.any() else np.full(n_resamples, np.nan)
    return out


def percentile_ci(replicates: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    if np.isnan(replicates).all():
        return float("nan"), float("nan")
    lo, hi = np.percentile(replicates, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)
