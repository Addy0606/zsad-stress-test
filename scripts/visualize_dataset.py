"""Save a grid of random test samples with ground-truth masks overlaid.

Usage:
    python scripts/visualize_dataset.py --dataset mvtec
    python scripts/visualize_dataset.py --dataset visa --paths kaggle --n 24 --out outputs/samples/visa.png
    python scripts/visualize_dataset.py --dataset mvtec --override categories=[screw]

Half the samples are anomalous (when available) so masks are actually visible.
Defect regions are tinted red with a red outline.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works on Kaggle and in tests)
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.registry import build_dataset, dataset_names  # noqa: E402


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Tint mask pixels red. image: H x W x 3 in [0, 1]; mask: H x W in {0, 1}."""
    out = image.copy()
    red = np.array([1.0, 0.0, 0.0])
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * red
    return out


def pick_indices(labels: list[int], n: int, seed: int) -> list[int]:
    """Up to n/2 anomalous and the rest normal, sampled with a fixed seed."""
    rng = random.Random(seed)
    anomalous = [i for i, y in enumerate(labels) if y == 1]
    normal = [i for i, y in enumerate(labels) if y == 0]
    n_anom = min(len(anomalous), n // 2)
    chosen = rng.sample(anomalous, n_anom) + rng.sample(normal, min(len(normal), n - n_anom))
    return sorted(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=dataset_names())
    p.add_argument("--paths", default="local", help="paths profile in configs/paths/ (local, kaggle)")
    p.add_argument("--override", nargs="*", default=[], help="config overrides, e.g. categories=[screw]")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: outputs/samples/<dataset>.png")
    a = p.parse_args()

    ds = build_dataset(a.dataset, a.paths, a.override)
    print(ds.summary())
    labels = [r.label for r in ds.records]
    idx = pick_indices(labels, a.n, a.seed)

    cols = 4
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.7))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    # There can be more grid cells than samples, so the lengths may differ on purpose.
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx, strict=False):
        item = ds[i]
        img = item["image"].permute(1, 2, 0).numpy()
        mask = item["mask"].numpy()
        ax.imshow(overlay(img, mask))
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="red", linewidths=0.8)
        tag = "normal" if item["label"] == 0 else (item["defect_type"] or "anomalous")
        if item["label"] == 1 and not ds.has_pixel_labels:
            tag += " (no mask)"
        ax.set_title(f"{item['category']}\n{tag[:32]}", fontsize=8)

    fig.suptitle(f"{ds.name}: {len(ds)} test images ({sum(labels)} anomalous), seed {a.seed}", fontsize=10)
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "outputs" / "samples" / f"{ds.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
