"""Tests for analysis/image_level.py: 5%-FPR thresholds, misses and the oracle."""

from __future__ import annotations

import numpy as np
import pytest

from analysis import image_level
from analysis.image_level import caught, complementarity


def test_caught_uses_normal_quantile_per_category():
    # category a: normals 0..99, so the 5%-FPR threshold is ~94; anomalies 50 (missed) and 99.5 (caught)
    cats = np.array(["a"] * 102)
    labels = np.r_[np.zeros(100, dtype=int), [1, 1]]
    scores = np.r_[np.arange(100.0), [50.0, 99.5]][None, :]
    hit = caught(scores, cats, labels, np.ones(102, dtype=int))
    assert not hit[0, 100] and hit[0, 101]
    assert hit[0, :100].mean() == pytest.approx(0.05, abs=0.011)


def test_oracle_catches_what_any_method_catches(monkeypatch):
    monkeypatch.setattr(image_level, "N_RESAMPLES", 20)
    n = 100
    labels = np.r_[np.zeros(n, dtype=int), np.ones(4, dtype=int)]
    rows_a, rows_b = [], []
    a_anom, b_anom = [200.0, 200.0, 0.0, 0.0], [200.0, 0.0, 200.0, 0.0]  # A misses 2,3; B misses 1,3
    for i, lab in enumerate(labels):
        sa = float(i) if lab == 0 else a_anom[i - n]
        sb = float(i) if lab == 0 else b_anom[i - n]
        base = {"category": "c", "label": int(lab), "defect_type": "good" if lab == 0 else "x", "key": f"{i}"}
        rows_a.append({**base, "score": sa})
        rows_b.append({**base, "score": sb})
    runs = {("A", "d"): rows_a, ("B", "d"): rows_b}
    res = {r["stat"]: r["value"] for r in complementarity(runs, ["A", "B"], "d")}
    assert res["miss:A"] == 0.5 and res["miss:B"] == 0.5
    assert res["miss:oracle_upper_bound"] == 0.25  # only image 3 is missed by both
    assert res["jaccard:A|B"] == pytest.approx(1 / 3)  # both miss {3}; either misses {1, 2, 3}


def test_or_rule_keeps_total_fpr():
    from analysis.image_level import or_rule

    rng = np.random.default_rng(0)
    n = 2000
    scores = rng.normal(size=(3, n))  # three independent detectors on normal images
    labels, cats, w = np.zeros(n, dtype=int), np.array(["c"] * n), np.ones(n, dtype=int)
    flagged = or_rule(scores, cats, labels, w, fpr=0.05)
    assert flagged.mean() == pytest.approx(0.05, abs=0.005)
    assert caught(scores, cats, labels, w).any(0).mean() > 0.12  # each at 5% -> union ~14%
