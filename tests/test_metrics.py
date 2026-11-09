"""Metric tests on toy inputs: compare against scikit-learn, against a slow brute-force AUPRO
written straight from the definition, and against hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from sklearn.metrics import f1_score, roc_auc_score

from eval.metrics import EIGHT_CONNECTED, aupro, category_metrics, image_auroc, image_f1_max, pixel_auroc


def brute_force_aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Definition-level AUPRO: loop over every distinct threshold. Slow, only for tests."""
    masks = masks.astype(bool)
    regions = []
    for i in range(len(masks)):
        lab, n = ndimage.label(masks[i], structure=EIGHT_CONNECTED)
        regions += [(i, lab == r) for r in range(1, n + 1)]
    normal = ~masks
    fprs, pros = [0.0], [0.0]
    for t in np.sort(np.unique(maps))[::-1]:
        above = maps >= t
        fprs.append((above & normal).sum() / normal.sum())
        pros.append(np.mean([above[i][reg].mean() for i, reg in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    f, p = fprs[keep], pros[keep]
    if f[-1] < fpr_limit:
        f, p = np.r_[f, fpr_limit], np.r_[p, np.interp(fpr_limit, fprs, pros)]
    return float(np.trapezoid(p, f) / fpr_limit)


def toy_data(seed: int, n: int = 6, size: int = 24):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), dtype=np.uint8)
    for i in range(n // 2):  # first half anomalous, with 1-2 regions of different sizes
        masks[i, 2:5, 2:5] = 1
        if i % 2 == 0:
            masks[i, 10:20, 12:22] = 1
    maps = rng.random((n, size, size)) + 0.8 * masks + 0.1 * rng.random((n, size, size)) * masks
    maps = np.round(maps, 2)  # create ties on purpose
    return masks, maps


def test_image_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 200), rng.random(200)
    assert image_auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_image_f1_max_is_best_threshold_f1():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 100), rng.random(100)
    best = max(f1_score(y, s >= t) for t in np.unique(s))
    assert image_f1_max(y, s) == pytest.approx(best)


def test_pixel_auroc_matches_sklearn():
    masks, maps = toy_data(2)
    assert pixel_auroc(masks, maps) == pytest.approx(roc_auc_score(masks.ravel(), maps.ravel()))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("limit", [0.3, 1.0, 0.05])
def test_aupro_matches_brute_force(seed, limit):
    masks, maps = toy_data(seed)
    assert aupro(masks, maps, limit) == pytest.approx(brute_force_aupro(masks, maps, limit), abs=1e-9)


def test_aupro_perfect_and_inverted():
    masks, _ = toy_data(0)
    assert aupro(masks, masks.astype(float)) == pytest.approx(1.0)  # defects score above every normal pixel
    assert aupro(masks, 1.0 - masks) == pytest.approx(0.0)  # defects score below every normal pixel


def test_aupro_weights_small_regions_equally():
    """One big region found, one small region missed -> PRO is 0.5, not ~1 (as pixel AUROC would suggest)."""
    masks = np.zeros((1, 20, 20), dtype=np.uint8)
    masks[0, 0:10, 0:10] = 1  # big region, 100 px
    masks[0, 15, 15] = 1  # small region, 1 px
    maps = np.full((1, 20, 20), 0.5)  # normal pixels
    maps[0, 0:10, 0:10] = 1.0  # big region: above every normal pixel
    maps[0, 15, 15] = 0.0  # small region: below every normal pixel, i.e. missed until FPR = 1
    # PRO is 0.5 for every FPR in [0, 1): flat curve, area / 0.3 = 0.5
    assert aupro(masks, maps, fpr_limit=0.3) == pytest.approx(0.5)
    # Pixel AUROC barely notices the miss: 100 of 101 defect pixels rank above all normal pixels.
    assert pixel_auroc(masks, maps) > 0.98


def test_category_metrics_skips_pixels_without_masks():
    rng = np.random.default_rng(4)
    out = category_metrics(rng.integers(0, 2, 20), rng.random(20), None, None)
    assert np.isnan(out["pixel_auroc"]) and np.isnan(out["aupro"])
    assert 0.0 <= out["image_auroc"] <= 1.0
