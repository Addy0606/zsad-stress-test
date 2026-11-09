"""Loader tests (spec, Phase 1): every loader returns the interface keys, masks are binary,
and normal images have empty masks.

Two layers:
1. Fake datasets (always run): tiny files in each real folder layout, built in tests/conftest.py.
2. Real data (run when present): the same checks on the local dev subset from configs/paths/local.yaml.
   Skipped automatically if a dataset isn't downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.base import ITEM_KEYS
from data.registry import LOADERS, build_dataset, load_dataset_config

ALL = sorted(LOADERS)


def check_item(item: dict, has_pixel_labels: bool) -> None:
    """The interface contract from the spec, checked on one item."""
    assert set(item) == set(ITEM_KEYS), f"keys differ: {set(item) ^ set(ITEM_KEYS)}"
    img, mask = item["image"], item["mask"]
    assert img.dtype == torch.float32 and img.ndim == 3 and img.shape[0] == 3
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert mask.dtype == torch.uint8 and mask.shape == img.shape[1:]
    assert set(mask.unique().tolist()) <= {0, 1}, "mask must be binary"
    assert item["label"] in (0, 1)
    assert item["domain"] in ("industrial", "medical")
    assert isinstance(item["category"], str) and isinstance(item["path"], str)
    if item["label"] == 0:
        assert mask.sum() == 0, "normal image must have an empty mask"
        assert item["defect_type"] is None
    else:
        assert isinstance(item["defect_type"], str)
        if has_pixel_labels:
            assert mask.sum() > 0, "anomalous image must have a non-empty mask"


# ---------- 1. Fake datasets ----------


@pytest.mark.parametrize("name", ALL)
def test_fake_dataset_items(name, fake_root):
    ds = LOADERS[name](root=fake_root(name))
    assert len(ds) > 0
    for i in range(len(ds)):
        check_item(ds[i], ds.has_pixel_labels)


@pytest.mark.parametrize("name", ALL)
def test_only_test_split_is_loaded(name, fake_root):
    ds = LOADERS[name](root=fake_root(name))
    for r in ds.records:
        assert "/train/" not in r.image_path.replace("\\", "/")
        assert "/valid/" not in r.image_path.replace("\\", "/")
        assert "0099" not in Path(r.image_path).name and "00098" not in Path(r.image_path).name


@pytest.mark.parametrize("name", ALL)
def test_both_labels_present(name, fake_root):
    labels = {r.label for r in LOADERS[name](root=fake_root(name)).records}
    assert labels == {0, 1}


@pytest.mark.parametrize("name", ALL)
def test_resize_keeps_mask_binary(name, fake_root):
    ds = LOADERS[name](root=fake_root(name), image_size=64)
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (3, 64, 64)
        check_item(item, ds.has_pixel_labels)


def test_category_filter(fake_root):
    ds = LOADERS["mvtec"](root=fake_root("mvtec"), categories=["screw"])
    assert ds.categories == ["screw"]


def test_visa_multi_defect_label(fake_root):
    ds = LOADERS["visa"](root=fake_root("visa"))
    defects = {r.defect_type for r in ds.records if r.label == 1}
    assert "scratch,dent" in defects  # quoted multi-defect labels are parsed as one string


def test_headct_has_no_pixel_labels(fake_root):
    ds = LOADERS["headct"](root=fake_root("headct"))
    assert ds.has_pixel_labels is False
    assert all(r.mask_path is None for r in ds.records)


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LOADERS["mvtec"](root=tmp_path / "does_not_exist")


def test_paths_are_unique(fake_root):
    for name in ALL:
        paths = [r.image_path for r in LOADERS[name](root=fake_root(name)).records]
        assert len(paths) == len(set(paths))


# ---------- 2. Real data (local dev subset), skipped if absent ----------


def _local_root(name: str) -> Path:
    return Path(load_dataset_config(name, "local").root)


@pytest.mark.parametrize("name", ALL)
def test_real_dataset_items(name):
    if not _local_root(name).exists():
        pytest.skip(f"{name} not downloaded locally")
    ds = build_dataset(name, "local")
    for i in range(len(ds)):
        check_item(ds[i], ds.has_pixel_labels)
