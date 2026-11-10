"""Per-region and per-image failure statistics (Phase 5, RQ1 and RQ4).

Everything is computed at the common evaluation size (256 x 256), from the cached maps.

Definitions
- Region: a connected (8-neighbour) blob of a ground-truth mask, as in AUPRO.
- Area: number of region pixels at 256 x 256 (65,536 pixels per image).
- Contrast: |mean grey level inside the region - mean grey level of a ring around it| (0-255 scale).
  The ring is the region grown by RING_WIDTH pixels, minus every defect pixel. NaN if the ring is empty.
- Pixel thresholds (per method and category): the score exceeded by 1% / 5% of the pixels of the
  category's *normal* images, i.e. a 1% / 5% pixel false-positive rate.
- Region overlap at a threshold: fraction of the region's pixels scoring at or above it
  (the PRO of that one region). A region counts as found at 1% FPR if its overlap is >= 0.5.
- Peak in mask (RQ4): whether the map's highest pixel lies inside the ground-truth mask; also checked with
  the mask grown by PEAK_TOLERANCE pixels, and the distance from the peak to the nearest defect pixel.

Usage (Kaggle CPU job, maps and datasets attached):
    python analysis/regions.py --out /kaggle/working/analysis
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.metrics import EIGHT_CONNECTED  # noqa: E402

RING_WIDTH = 5
PEAK_TOLERANCE = 4
FPRS = (0.01, 0.05)
FOUND_OVERLAP = 0.5


def label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=EIGHT_CONNECTED)


def region_geometry(mask: np.ndarray, gray: np.ndarray) -> list[dict]:
    """Area and contrast of every region of one image. mask, gray: (H, W)."""
    labelled, n = label_regions(mask)
    defect = labelled > 0
    out = []
    for r in range(1, n + 1):
        region = labelled == r
        ring = ndimage.binary_dilation(region, structure=EIGHT_CONNECTED, iterations=RING_WIDTH) & ~defect
        contrast = abs(gray[region].mean() - gray[ring].mean()) if ring.any() else float("nan")
        out.append({"region": r, "area": int(region.sum()), "contrast": float(contrast)})
    return out


def normal_thresholds(normal_maps: np.ndarray, fprs: tuple[float, ...] = FPRS) -> dict[float, float]:
    """Pixel score exceeded by a fraction `fpr` of the normal-image pixels, for each fpr."""
    flat = normal_maps.ravel()
    return {f: float(np.quantile(flat, 1 - f)) for f in fprs}


def region_scores(amap: np.ndarray, mask: np.ndarray, thresholds: dict[float, float]) -> list[dict]:
    """Peak score and overlap at each threshold, for every region of one image."""
    labelled, n = label_regions(mask)
    out = []
    for r in range(1, n + 1):
        vals = amap[labelled == r]
        row = {"region": r, "max_score": float(vals.max())}
        for f, t in thresholds.items():
            row[f"overlap_fpr{int(round(100 * f))}"] = float((vals >= t).mean())
        out.append(row)
    return out


def peak_in_mask(amap: np.ndarray, mask: np.ndarray, tolerance: int = PEAK_TOLERANCE) -> dict:
    """Is the map's highest pixel inside the mask (exactly / within `tolerance` px)? Distance in px."""
    mask = mask.astype(bool)
    y, x = np.unravel_index(np.argmax(amap), amap.shape)
    grown = ndimage.binary_dilation(mask, structure=EIGHT_CONNECTED, iterations=tolerance) if tolerance else mask
    dist = ndimage.distance_transform_edt(~mask)[y, x]
    return {"peak_inside": bool(mask[y, x]), "peak_inside_tol": bool(grown[y, x]), "peak_dist": float(dist)}


def gray_at(path: str, size: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dataset_geometry(dataset, size: int) -> list[dict]:
    """Region table of a dataset (independent of any method). Paths relative to the dataset root."""
    from eval.cache import relative_path

    rows = []
    for i, rec in enumerate(dataset.records):
        if not rec.label:
            continue
        mask = dataset.load_mask(i, size)
        if not mask.any():
            continue
        rel = relative_path(rec.image_path, dataset.root)
        for g in region_geometry(mask, gray_at(rec.image_path, size)):
            rows.append({"image": rel, "category": rec.category, "defect_type": rec.defect_type, **g})
    return rows


def method_tables(dataset, rdir: Path, size: int) -> tuple[list[dict], list[dict], list[dict]]:
    """(region rows, anomalous-image rows, threshold rows) for one method's run on one dataset."""
    from eval.cache import load_map, match_records, read_scores, relative_path

    rows = read_scores(rdir)
    index_of = match_records(dataset, rdir, rows)
    regions, images, thresholds = [], [], []
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        normal = [r for r in cat_rows if r["label"] == 0]
        th = normal_thresholds(np.stack([load_map(rdir, r["path"]) for r in normal]))
        thresholds.append({"category": cat, **{f"thr_fpr{int(round(100 * f))}": t for f, t in th.items()}})
        for r in cat_rows:
            if r["label"] == 0:
                continue
            idx = index_of[r["path"]]
            mask = dataset.load_mask(idx, size)
            if not mask.any():
                continue
            amap = load_map(rdir, r["path"])
            rel = relative_path(dataset.records[idx].image_path, dataset.root)
            for s in region_scores(amap, mask, th):
                regions.append({"image": rel, "category": cat, **s})
            images.append({"image": rel, "category": cat, "score": r["score"], **peak_in_mask(amap, mask)})
    return regions, images, thresholds


def main() -> None:
    from omegaconf import OmegaConf

    from data.registry import build_dataset
    from eval.cache import EVAL_SIZE

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/analysis")
    p.add_argument("--paths", default="kaggle")
    p.add_argument("--datasets", nargs="*", default=["mvtec", "visa", "mpdd", "bmad_brain"])
    a = p.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    from bench_stats import find_run

    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    out, failed = Path(a.out), []
    for dataset_name in a.datasets:
        dataset = build_dataset(dataset_name, a.paths)
        print(f"===== geometry {dataset_name}", flush=True)
        write_csv(out / "geometry" / f"{dataset_name}.csv", dataset_geometry(dataset, EVAL_SIZE))
        for method, by_dataset in manifest.runs.items():
            src = by_dataset[dataset_name]
            rdir = find_run(Path(a.input), src.kernel, src.run)
            print(f"===== {method} on {dataset_name}: {rdir}", flush=True)
            try:
                regions, images, thresholds = method_tables(dataset, rdir, EVAL_SIZE)
                write_csv(out / method / f"{dataset_name}_regions.csv", regions)
                write_csv(out / method / f"{dataset_name}_images.csv", images)
                write_csv(out / method / f"{dataset_name}_thresholds.csv", thresholds)
            except Exception as e:
                traceback.print_exc()
                failed.append((method, dataset_name, repr(e)))
    print("FAILED:", failed if failed else "none")


if __name__ == "__main__":
    main()
