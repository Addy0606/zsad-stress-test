"""Tests for eval/pixel_stats.py (binned pixel metrics) and eval/bootstrap.py (CIs)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.bootstrap import bootstrap_metrics, percentile_ci, resample_counts
from eval.metrics import aupro, image_auroc, pixel_auroc
from eval.pixel_stats import binned_pixel_auroc, compute_stats, stats_metrics


def fake_category(n: int = 12, size: int = 48, seed: int = 0):
    """Half the images get 1-2 square defects; maps = noise + a bump on defects (imperfect detector)."""
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), dtype=bool)
    for i in range(n // 2, n):
        for _ in range(rng.integers(1, 3)):
            y, x, s = rng.integers(0, size - 12), rng.integers(0, size - 12), rng.integers(2, 12)
            masks[i, y : y + s, x : x + s] = True
    maps = rng.normal(size=masks.shape) + 1.5 * masks
    labels = masks.reshape(n, -1).any(1).astype(int)
    scores = maps.reshape(n, -1).max(1)
    return masks, maps, labels, scores


def test_binned_matches_exact():
    masks, maps, _, _ = fake_category()
    s = compute_stats(masks, maps)
    p_auroc, p_aupro = stats_metrics(s)
    assert p_auroc == pytest.approx(pixel_auroc(masks, maps), abs=2e-3)
    assert p_aupro == pytest.approx(aupro(masks, maps), abs=5e-3)


def test_binned_auroc_counts_same_bin_as_tie():
    # one bin each: normal pixels below, defect pixels above -> perfect; all in one bin -> 0.5
    assert binned_pixel_auroc(np.array([5, 0]), np.array([0, 3])) == 1.0
    assert binned_pixel_auroc(np.array([5]), np.array([3])) == 0.5


def test_image_weights_equal_duplicating_images():
    """Weighting image i by k must give the metric of the data with image i repeated k times."""
    masks, maps, _, _ = fake_category(seed=1)
    s = compute_stats(masks, maps)
    w = np.array([2, 0, 1, 1, 3, 0, 1, 2, 0, 1, 1, 2])
    idx = np.repeat(np.arange(len(w)), w)
    dup = {k: s[k] for k in ("edges",)}
    dup["neg"], dup["pos"] = s["neg"][idx], s["pos"][idx]
    keep = np.concatenate([np.flatnonzero(s["reg_img"] == i) for i in idx])
    dup["reg"], dup["reg_img"] = (
        s["reg"][keep],
        np.repeat(np.arange(len(idx)), [np.sum(s["reg_img"] == i) for i in idx]),
    )
    assert stats_metrics(s, w) == pytest.approx(stats_metrics(dup))


def test_resample_counts_keeps_class_sizes():
    labels = np.array([0] * 7 + [1] * 5)
    counts = resample_counts(labels, np.random.default_rng(0))
    assert counts[labels == 0].sum() == 7 and counts[labels == 1].sum() == 5


def _rows(categories):
    rows = []
    for cat, (labels, scores) in categories.items():
        for i, (lab, sc) in enumerate(zip(labels, scores, strict=True)):
            rows.append({"category": cat, "label": int(lab), "score": float(sc), "path": f"{cat}/{i}.png"})
    return rows


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    cats, stats = {}, {}
    for k, cat in enumerate(["a", "b"]):
        masks, maps, labels, scores = fake_category(n=20, seed=10 + k)
        cats[cat] = (labels, scores)
        stats[cat] = {**compute_stats(masks, maps), "paths": np.array([f"{cat}/{i}.png" for i in range(20)])}
    rows = _rows(cats)
    reps = bootstrap_metrics(rows, stats, n_resamples=200, seed=0)
    again = bootstrap_metrics(rows, stats, n_resamples=200, seed=0)
    for m in ("image_auroc", "pixel_auroc", "aupro"):
        np.testing.assert_array_equal(reps[m], again[m])
    point = np.mean([image_auroc(*cats[c]) for c in cats])
    lo, hi = percentile_ci(reps["image_auroc"])
    assert lo <= point <= hi and hi - lo > 0
    lo, hi = percentile_ci(reps["aupro"])
    point = np.mean([stats_metrics(stats[c])[1] for c in stats])
    assert lo <= point <= hi


def test_bootstrap_stats_rows_matched_by_path_not_order():
    """Pixel stats may be stored in a different image order than scores.csv."""
    masks, maps, labels, scores = fake_category(n=20, seed=3)
    s = {**compute_stats(masks, maps), "paths": np.array([f"a/{i}.png" for i in range(20)])}
    rows = _rows({"a": (labels, scores)})
    shuffled = list(reversed(rows))
    a = bootstrap_metrics(rows, {"a": s}, n_resamples=50, seed=0)
    b = bootstrap_metrics(shuffled, {"a": s}, n_resamples=50, seed=0)
    # same multiset of images, different order -> different draws, but both close to the point value
    point = stats_metrics(s)[0]
    assert abs(np.median(a["pixel_auroc"]) - point) < 0.02 and abs(np.median(b["pixel_auroc"]) - point) < 0.02


def test_perfect_detector_has_zero_width_ci():
    labels = np.array([0] * 10 + [1] * 10)
    reps = bootstrap_metrics(_rows({"a": (labels, labels.astype(float))}), None, n_resamples=100)
    assert percentile_ci(reps["image_auroc"]) == (1.0, 1.0)
    assert np.isnan(reps["aupro"]).all()
