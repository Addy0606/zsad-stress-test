"""End-to-end check of eval/evaluate.py on a fake dataset: a cache of perfect maps must score 1.0,
at the default map size and at a custom --eval-size."""

from __future__ import annotations

import json

import numpy as np
import pytest

from data.registry import LOADERS
from eval.cache import EVAL_SIZE, append_score, save_map
from eval.evaluate import evaluate_run


@pytest.mark.parametrize("eval_size", [EVAL_SIZE, 64])
def test_perfect_cache_scores_one(eval_size, fake_root, tmp_path):
    root = fake_root("mvtec")
    overrides = [f"root={root.as_posix()}", "categories=[bottle,screw]"]
    ds = LOADERS["mvtec"](root=root)
    rdir = tmp_path / "run"
    for i in range(len(ds)):
        item = ds[i]
        save_map(rdir, item["path"], ds.load_mask(i, eval_size).astype(np.float32))  # map == ground truth
        append_score(rdir, {**item, "score": float(item["label"]), "seconds": 0.0})

    result = evaluate_run(rdir, "mvtec", "local", overrides, eval_size=eval_size)

    assert result["eval_size"] == eval_size
    assert result["complete"]
    assert result["n_categories"] == 2
    for k in ("image_auroc", "image_f1_max", "pixel_auroc", "aupro"):
        assert result["mean"][k] == pytest.approx(1.0), k
    assert json.loads((rdir / "metrics.json").read_text())["eval_size"] == eval_size


def test_eval_size_mismatch_fails(fake_root, tmp_path):
    """Maps saved at one size and evaluated at another must not silently pass."""
    root = fake_root("mvtec")
    ds = LOADERS["mvtec"](root=root)
    rdir = tmp_path / "run"
    for i in range(len(ds)):
        item = ds[i]
        save_map(rdir, item["path"], ds.load_mask(i, 64).astype(np.float32))
        append_score(rdir, {**item, "score": float(item["label"]), "seconds": 0.0})
    with pytest.raises(ValueError):
        evaluate_run(rdir, "mvtec", "local", [f"root={root.as_posix()}", "categories=[bottle,screw]"], eval_size=32)
