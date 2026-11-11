"""Failure galleries (spec Phase 5): for each detector and dataset, its 10 worst missed anomalies and 10 worst
false alarms, each shown as the photo with the ground-truth defect outlined, and the detector's anomaly map.

"Worst" is measured inside each image's category, because scores are only comparable within a category:
    z = (score - mean of the category's normal scores) / their standard deviation
- worst missed anomalies: anomalous images with the lowest z (they look most normal to the detector)
- worst false alarms:     normal images with the highest z (they look most anomalous)
Maps are coloured on the demo's common scale: each value through the detector's training-data normaliser
(app/fusion_normalizer.npz), so blue = ordinary for that detector, red = higher than nearly all its training pixels.

Runs as a Kaggle CPU job with the benchmark outputs and the datasets attached (the maps and images live there):
    python analysis/galleries.py --out /kaggle/working/galleries
Writes <out>/<method>/<dataset>.jpg and <out>/galleries.csv (every image shown, with its score, z and percentile).
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from eval.cache import EVAL_SIZE, read_scores, relative_path, run_dataset_root  # noqa: E402
from fusion.normalize import QuantileNormalizer  # noqa: E402
from fusion.pixel_fusion import DETECTORS, DatasetMaps  # noqa: E402

DATASETS = ("mvtec", "visa", "mpdd", "bmad_brain", "headct")
NAMES = {
    "anomalyclip": "AnomalyCLIP",
    "visualad": "VisualAD",
    "anomalyvfm_dinov2": "AnomalyVFM (DINOv2)",
    "anomalyvfm_radio": "AnomalyVFM (RADIO)",
    "mvtec": "MVTec AD",
    "visa": "VisA",
    "mpdd": "MPDD",
    "bmad_brain": "BMAD Brain MRI",
    "headct": "Head CT",
}
N_SHOW = 10
COLORS = np.array([[0, 0, 130], [0, 120, 255], [0, 230, 230], [255, 230, 0], [230, 0, 0]], dtype=np.float32) / 255


def load_normalizer() -> QuantileNormalizer:
    z = np.load(REPO / "app" / "fusion_normalizer.npz")
    assert tuple(z["detectors"]) == DETECTORS
    norm = QuantileNormalizer(n_quantiles=len(z["grid"]))
    norm.grid_ = z["grid"]
    return norm


def heat(values: np.ndarray) -> np.ndarray:
    """Values in [0, 1] -> RGB with the demo's blue-to-red scale."""
    v = values.clip(0, 1) * (len(COLORS) - 1)
    lo = np.floor(v).astype(int).clip(0, len(COLORS) - 2)
    f = (v - lo)[..., None]
    return COLORS[lo] * (1 - f) + COLORS[lo + 1] * f


def scores_by_key(d: DatasetMaps, j: int) -> dict[str, float]:
    rdir = next(iter(d.sources[j].values()))[0]
    root = run_dataset_root(rdir)
    return {relative_path(r["path"], root): r["score"] for r in read_scores(rdir)}


def z_scores(score: np.ndarray, cats: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per image: z against its category's normal scores, and the % of those normal scores strictly below it."""
    z, pct = np.zeros(len(score)), np.zeros(len(score))
    for c in np.unique(cats):
        idx = cats == c
        normal = score[idx & (labels == 0)]
        z[idx] = (score[idx] - normal.mean()) / (normal.std() + 1e-12)
        pct[idx] = 100 * (normal[None, :] < score[idx][:, None]).mean(1)
    return z, pct


def pick(z: np.ndarray, labels: np.ndarray, keys: list[str]) -> dict[str, list[int]]:
    anom, norm = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    return {
        "missed": sorted(anom, key=lambda i: (z[i], keys[i]))[:N_SHOW],
        "false_alarm": sorted(norm, key=lambda i: (-z[i], keys[i]))[:N_SHOW],
    }


def photo(d: DatasetMaps, i: int) -> np.ndarray:
    rec = d.dataset.records[d.index[i]]
    img = Image.open(rec.image_path).convert("RGB").resize((EVAL_SIZE, EVAL_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255


def draw(d: DatasetMaps, j: int, chosen: dict, z: np.ndarray, pct: np.ndarray, norm: QuantileNormalizer, out: Path):
    m = DETECTORS[j]
    # rows: photo, map | blank spacer | photo, map
    fig, axes = plt.subplots(
        5, N_SHOW, figsize=(1.45 * N_SHOW, 7.6), facecolor="white", gridspec_kw={"height_ratios": [1, 1, 0.45, 1, 1]}
    )
    for ax in axes[2]:
        ax.axis("off")
    blocks = (
        ("missed", "10 worst missed anomalies (lowest score vs. normal images of their category)", 0),
        ("false_alarm", "10 worst false alarms (normal images scored highest vs. their category)", 3),
    )
    for kind, title, row in blocks:
        for k in range(N_SHOW):
            top, bottom = axes[row, k], axes[row + 1, k]
            for ax in (top, bottom):
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            if k >= len(chosen[kind]):
                top.axis("off")
                bottom.axis("off")
                continue
            i = chosen[kind][k]
            img = photo(d, i)
            top.imshow(img)
            if kind == "missed" and d.has_masks:
                top.contour(d.mask(i).astype(float), levels=[0.5], colors="red", linewidths=1.2)
            rec = d.dataset.records[d.index[i]]
            label = rec.defect_type or "normal"
            top.set_title(f"{rec.category[:14]}\n{label[:16]}", fontsize=6.5, pad=2)
            cols = np.zeros((EVAL_SIZE * EVAL_SIZE, len(DETECTORS)))  # the normaliser works on all four columns;
            cols[:, j] = d.maps(i)[j].ravel()  # only this detector's column is used
            mp = norm.transform(cols)[:, j]
            bottom.imshow(0.45 * img + 0.55 * heat(mp.reshape(EVAL_SIZE, EVAL_SIZE)))
            bottom.set_xlabel(f"z {z[i]:+.1f} · {pct[i]:.0f}%", fontsize=6.5, labelpad=2)
        axes[row, 0].annotate(title, (0, 1.42), xycoords="axes fraction", fontsize=9, weight="bold", ha="left")
    fig.suptitle(f"{NAMES[m]} on {NAMES[d.name]}", x=0.01, y=0.995, ha="left", fontsize=11, weight="bold")
    fig.text(
        0.01,
        0.005,
        "Top row of each block: photo (red outline = ground-truth defect). Bottom row: the detector's map "
        "(blue = ordinary, red = higher than nearly all its training pixels). "
        "z = score vs. normal images of the category; % = share of those normal images scoring lower.",
        fontsize=6.5,
        color="#52514e",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.05, hspace=0.5, wspace=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=85, pil_kwargs={"quality": 85})
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/galleries")
    p.add_argument("--paths", default="kaggle")
    a = p.parse_args()
    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    norm = load_normalizer()
    out = Path(a.out)
    rows, failed = [], []
    for ds in DATASETS:
        print(f"===== {ds}", flush=True)
        try:
            d = DatasetMaps(ds, a.paths, Path(a.input), manifest)
            for j, m in enumerate(DETECTORS):
                by_key = scores_by_key(d, j)
                score = np.array([by_key[k] for k in d.keys])
                z, pct = z_scores(score, d.cats, d.labels)
                chosen = pick(z, d.labels, d.keys)
                draw(d, j, chosen, z, pct, norm, out / m / f"{ds}.jpg")
                for kind, idx in chosen.items():
                    for rank, i in enumerate(idx, 1):
                        rec = d.dataset.records[d.index[i]]
                        rows.append(
                            {
                                "method": m,
                                "dataset": ds,
                                "kind": kind,
                                "rank": rank,
                                "image": d.keys[i],
                                "category": rec.category,
                                "defect_type": rec.defect_type or "",
                                "score": round(float(score[i]), 6),
                                "z": round(float(z[i]), 3),
                                "pct_normals_below": round(float(pct[i]), 1),
                            }
                        )
                print(f"   {m}: done", flush=True)
        except Exception as e:
            traceback.print_exc()
            failed.append((ds, repr(e)))
    with open(out / "galleries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("FAILED:", failed if failed else "none")


if __name__ == "__main__":
    main()
