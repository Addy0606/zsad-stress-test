"""Read and write the per-run cache. Analysis and fusion read ONLY these files.

    outputs/<method>/<dataset>/
        scores.csv         one row per image: path, label, score, category, defect_type, domain, dataset, seconds
        maps/<id>.npz      anomaly map, EVAL_SIZE x EVAL_SIZE, float16, compressed
        config.yaml        resolved method + dataset config, seeds, versions, git commit
        run_info.json      runtime per image, peak GPU memory, timestamps
        metrics.json       written by eval/evaluate.py (per category + mean)

A row is appended to scores.csv only after its map is saved, so a row always has a map.
That makes every run resumable: on restart, images already in scores.csv are skipped.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

EVAL_SIZE = 256  # every map (and mask) is compared at 256 x 256 so methods are on equal footing
SCORE_FIELDS = ["path", "label", "score", "category", "defect_type", "domain", "dataset", "seconds"]


def run_dir(out_root: str | Path, method: str, dataset: str) -> Path:
    return Path(out_root) / method / dataset


def map_id(path: str) -> str:
    """Stable, filesystem-safe file name for an image path: '<parent>_<stem>_<short hash>'."""
    p = Path(path)
    digest = hashlib.sha1(p.as_posix().encode()).hexdigest()[:10]
    return f"{p.parent.name}_{p.stem}_{digest}"


def done_paths(rdir: Path) -> set[str]:
    f = rdir / "scores.csv"
    if not f.exists():
        return set()
    with open(f, newline="", encoding="utf-8") as fh:
        return {row["path"] for row in csv.DictReader(fh)}


def save_map(rdir: Path, path: str, amap: np.ndarray) -> None:
    (rdir / "maps").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rdir / "maps" / f"{map_id(path)}.npz", amap=amap.astype(np.float16))


def load_map(rdir: Path, path: str) -> np.ndarray:
    return np.load(rdir / "maps" / f"{map_id(path)}.npz")["amap"].astype(np.float32)


def append_score(rdir: Path, row: dict) -> None:
    f = rdir / "scores.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in SCORE_FIELDS})
        fh.flush()


def read_scores(rdir: Path) -> list[dict]:
    with open(rdir / "scores.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["label"], r["score"] = int(r["label"]), float(r["score"])
        r["defect_type"] = r["defect_type"] or None
    return rows


def relative_path(path: str, root: str | Path) -> str:
    """`path` relative to `root`, with forward slashes. Image paths in scores.csv are absolute, and the
    dataset root can differ between sessions (Kaggle has mounted the same dataset at different
    folders), so images are matched across runs by this relative path."""
    p, r = str(path).replace("\\", "/"), str(root).replace("\\", "/").rstrip("/")
    if not p.startswith(r + "/"):
        raise ValueError(f"{path} is not under {root}")
    return p[len(r) + 1 :]


def run_dataset_root(rdir: Path) -> str | None:
    """The dataset root a run used, from its config.yaml (None if the run has no config.yaml)."""
    from omegaconf import OmegaConf

    cfg = Path(rdir) / "config.yaml"
    return str(OmegaConf.load(cfg).dataset.root) if cfg.exists() else None


def match_records(dataset, rdir: Path, rows: list[dict]) -> dict[str, int]:
    """{scores.csv path -> index in `dataset`}, matching by path relative to each side's dataset root."""
    index = {relative_path(r.image_path, dataset.root): i for i, r in enumerate(dataset.records)}
    root = run_dataset_root(rdir) or dataset.root  # no config.yaml: assume the same root
    return {row["path"]: index[relative_path(row["path"], root)] for row in rows}
