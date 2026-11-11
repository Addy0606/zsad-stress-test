"""Tests for how the failure galleries rank images (analysis/galleries.py)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.galleries import pick, z_scores


def test_z_scores_are_relative_to_each_categorys_normals():
    # category a: normals 1, 2, 3; category b: normals 10, 20, 30 (same shape, 10x the scale)
    score = np.array([1, 2, 3, 2.5, 10, 20, 30, 25.0])
    cats = np.array(["a"] * 4 + ["b"] * 4)
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    z, pct = z_scores(score, cats, labels)
    np.testing.assert_allclose(z[3], z[7])  # same position relative to their own normals
    assert pct[3] == pytest.approx(pct[7]) == pytest.approx(100 * 2 / 3)  # two of three normals score lower


def test_pick_takes_lowest_anomalies_and_highest_normals():
    z = np.array([0.5, -2.0, 3.0, 1.0, -1.0, 2.0])
    labels = np.array([1, 1, 0, 0, 1, 0])
    keys = [f"img{i}" for i in range(6)]
    chosen = pick(z, labels, keys)
    assert chosen["missed"][:3] == [1, 4, 0]  # anomalies, most normal-looking first
    assert chosen["false_alarm"][:3] == [2, 5, 3]  # normals, most anomalous-looking first
