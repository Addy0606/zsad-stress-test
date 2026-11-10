"""Tests for analysis/regions.py: region geometry, thresholds, region overlap, peak-in-mask."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.regions import normal_thresholds, peak_in_mask, region_geometry, region_scores


def two_region_mask() -> np.ndarray:
    m = np.zeros((40, 40), dtype=np.uint8)
    m[5:10, 5:10] = 1  # 25 px
    m[20:30, 20:22] = 1  # 20 px
    return m


def test_geometry_area_and_contrast():
    mask = two_region_mask()
    gray = np.full((40, 40), 100.0)
    gray[5:10, 5:10] = 160.0  # bright defect: contrast 60
    gray[20:30, 20:22] = 100.0  # invisible defect: contrast 0
    g = region_geometry(mask, gray)
    assert [r["area"] for r in g] == [25, 20]
    assert g[0]["contrast"] == pytest.approx(60.0)
    assert g[1]["contrast"] == pytest.approx(0.0)


def test_ring_excludes_other_defect_pixels():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:8] = 1
    mask[5:8, 9:12] = 1  # second region 1 px away: its pixels must not count as background
    gray = np.full((20, 20), 50.0)
    gray[mask.astype(bool)] = 200.0
    g = region_geometry(mask, gray)
    assert g[0]["contrast"] == pytest.approx(150.0)


def test_normal_thresholds_are_fpr_quantiles():
    maps = np.arange(10_000, dtype=np.float64).reshape(1, 100, 100)
    th = normal_thresholds(maps, fprs=(0.01, 0.05))
    assert (maps >= th[0.01]).mean() == pytest.approx(0.01, abs=1e-4)
    assert (maps >= th[0.05]).mean() == pytest.approx(0.05, abs=1e-4)


def test_region_overlap_and_peak():
    mask = two_region_mask()
    amap = np.zeros((40, 40))
    amap[5:10, 5:10] = 1.0  # first region fully above threshold
    amap[20:25, 20:22] = 1.0  # half of the second
    s = region_scores(amap, mask, {0.01: 0.5})
    assert s[0]["overlap_fpr1"] == 1.0 and s[1]["overlap_fpr1"] == 0.5
    assert s[0]["max_score"] == 1.0


def test_peak_in_mask_inside_near_and_far():
    mask = two_region_mask()
    amap = np.zeros((40, 40))
    amap[7, 7] = 1.0
    assert peak_in_mask(amap, mask) == {"peak_inside": True, "peak_inside_tol": True, "peak_dist": 0.0}
    amap[:] = 0
    amap[7, 12] = 1.0  # 3 px right of the first region (last column 9)
    r = peak_in_mask(amap, mask, tolerance=4)
    assert not r["peak_inside"] and r["peak_inside_tol"] and r["peak_dist"] == pytest.approx(3.0)
    amap[:] = 0
    amap[38, 2] = 1.0  # far corner
    r = peak_in_mask(amap, mask, tolerance=4)
    assert not r["peak_inside"] and not r["peak_inside_tol"] and r["peak_dist"] > 10
