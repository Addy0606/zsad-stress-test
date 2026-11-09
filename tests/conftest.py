"""Tiny fake datasets in each real folder layout, so loader tests run without any downloads.

Each builder writes a few 32x40 images (deliberately non-square, to catch H/W mix-ups) and
returns the root folder. Masks are written as 0/255 (MVTec, MPDD, BMAD) or 0/k (VisA), like the
real datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H, W = 32, 40


def _img(path: Path, gray: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(str(path))) % 2**32)
    arr = rng.integers(0, 255, size=(H, W) if gray else (H, W, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((H, W), dtype=np.uint8)
    m[8:16, 10:20] = value
    Image.fromarray(m).save(path)


def make_mvtec_layout(root: Path) -> Path:
    for cat, defects in {"bottle": ["broken_large"], "screw": ["scratch_head", "thread_top"]}.items():
        for i in range(3):
            _img(root / cat / "test" / "good" / f"{i:03d}.png")
        for d in defects:
            for i in range(2):
                _img(root / cat / "test" / d / f"{i:03d}.png")
                _mask(root / cat / "ground_truth" / d / f"{i:03d}_mask.png")
        _img(root / cat / "train" / "good" / "000.png")  # train split must be ignored
    return root


def make_visa_layout(root: Path) -> Path:
    rows = ["object,split,label,image,mask"]
    for cat in ["candle", "pcb1"]:
        anno = ["image,label,mask"]
        for i in range(3):
            rel = f"{cat}/Data/Images/Normal/{i:04d}.JPG"
            _img(root / rel)
            rows.append(f"{cat},test,normal,{rel},")
            anno.append(f"{rel},normal,")
        _img(root / f"{cat}/Data/Images/Normal/0099.JPG")
        rows.append(f"{cat},train,normal,{cat}/Data/Images/Normal/0099.JPG,")  # train row must be ignored
        for i in range(2):
            rel, mrel = f"{cat}/Data/Images/Anomaly/{i:03d}.JPG", f"{cat}/Data/Masks/Anomaly/{i:03d}.png"
            _img(root / rel)
            _mask(root / mrel, value=i + 1)  # VisA masks use small integer values
            rows.append(f"{cat},test,anomaly,{rel},{mrel}")
            label = '"scratch,dent"' if i == 0 else "scratch"  # multi-defect labels are quoted
            anno.append(f"{rel},{label},{mrel}")
        (root / cat).mkdir(parents=True, exist_ok=True)
        (root / cat / "image_anno.csv").write_text("\n".join(anno) + "\n")
    (root / "split_csv").mkdir(parents=True, exist_ok=True)
    (root / "split_csv" / "1cls.csv").write_text("\n".join(rows) + "\n")
    return root


def make_headct_layout(root: Path) -> Path:
    lines = ["id, hemorrhage"]
    for i in range(6):
        _img(root / "head_ct" / "head_ct" / f"{i:03d}.png", gray=True)
        lines.append(f"{i},{1 if i < 3 else 0}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "labels.csv").write_text("\n".join(lines) + "\n")
    return root


def make_bmad_layout(root: Path) -> Path:
    for i in range(3):
        _img(root / "test" / "good" / "img" / f"{i:05d}_60.png", gray=True)
        _img(root / "test" / "Ungood" / "img" / f"{i + 10:05d}_60.png", gray=True)
        _mask(root / "test" / "Ungood" / "label" / f"{i + 10:05d}_60.png")
    _img(root / "train" / "good" / "00099_60.png", gray=True)  # train split must be ignored
    _img(root / "valid" / "good" / "img" / "00098_60.png", gray=True)  # valid split must be ignored
    return root


BUILDERS = {
    "mvtec": make_mvtec_layout,
    "mpdd": make_mvtec_layout,
    "visa": make_visa_layout,
    "headct": make_headct_layout,
    "bmad_brain": make_bmad_layout,
}


@pytest.fixture
def fake_root(tmp_path):
    """Returns a function: fake_root(name) -> root folder of a fake dataset with that name's layout."""

    def _make(name: str) -> Path:
        return BUILDERS[name](tmp_path / name)

    return _make
