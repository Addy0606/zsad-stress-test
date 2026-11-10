"""Per-image pixel statistics, so pixel AUROC and AUPRO can be bootstrapped without the maps.

Bootstrapping pixel metrics directly means recomputing them over millions of pixels 1,000 times,
and needs every anomaly map (gigabytes). Instead, for each category we fix one set of score bins
(quantiles of that category's pixel scores) and store, per image:

    neg[i, b]   number of normal pixels of image i in bin b
    pos[i, b]   number of defect pixels of image i in bin b
and per ground-truth defect region r (8-connected, as in AUPRO):
    reg[r, b]   number of pixels of region r in bin b,   reg_img[r] = the image it belongs to

Both metrics are sums over images, so any resample of images is just a weighted sum of rows:
    pixel AUROC = P(defect pixel scores higher than normal pixel), ties within a bin count 1/2
    AUPRO       = area under (mean over regions of the fraction above t) vs (fraction of normal
                  pixels above t), for t = each bin edge, up to FPR 0.3, divided by 0.3
Binning makes these approximate; `validate` reports the difference from the exact metrics.
Headline numbers always come from the exact metrics in eval/metrics.py; bins are only for CIs.

Usage (on Kaggle, where the maps live):
    python eval/pixel_stats.py --run <dir with scores.csv + maps/> --dataset mvtec --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import EIGHT_CONNECTED, aupro, pixel_auroc  # noqa: E402

N_BINS = 1000


def bin_edges(maps: np.ndarray, n_bins: int = N_BINS, sample: int = 2_000_000, seed: int = 0) -> np.ndarray:
    """Interior bin edges at quantiles of the pixel scores (a random sample of them, for speed)."""
    flat = maps.ravel()
    if flat.size > sample:
        flat = np.random.default_rng(seed).choice(flat, sample, replace=False)
    return np.unique(np.quantile(flat, np.linspace(0, 1, n_bins + 1)[1:-1]))


def compute_stats(masks: np.ndarray, maps: np.ndarray, n_bins: int = N_BINS) -> dict[str, np.ndarray]:
    """Histograms for one category. masks: (N, H, W) {0,1}; maps: (N, H, W) float."""
    edges = bin_edges(maps, n_bins)
    nb = len(edges) + 1
    masks = masks.astype(bool)
    neg = np.zeros((len(maps), nb), dtype=np.int64)
    pos = np.zeros((len(maps), nb), dtype=np.int64)
    reg, reg_img = [], []
    for i in range(len(maps)):
        bins = np.searchsorted(edges, maps[i], side="right")  # bin index of every pixel, 0..nb-1
        neg[i] = np.bincount(bins[~masks[i]], minlength=nb)
        pos[i] = np.bincount(bins[masks[i]], minlength=nb)
        labelled, n = ndimage.label(masks[i], structure=EIGHT_CONNECTED)
        for r in range(1, n + 1):
            reg.append(np.bincount(bins[labelled == r], minlength=nb))
            reg_img.append(i)
    reg_arr = np.array(reg, dtype=np.int64).reshape(-1, nb)
    return {"edges": edges, "neg": neg, "pos": pos, "reg": reg_arr, "reg_img": np.array(reg_img, dtype=np.int64)}


def binned_pixel_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """AUROC from per-bin counts (1-D, summed over images). Same-bin pairs count as ties."""
    neg, pos = neg.astype(np.float64), pos.astype(np.float64)
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def region_frac_above(reg: np.ndarray) -> np.ndarray:
    """(R, B): for each region, the fraction of its pixels at or above each bin boundary, top bin first.
    It doesn't depend on which images are resampled, so the bootstrap computes it once."""
    return np.cumsum(reg[:, ::-1], axis=1) / reg.sum(axis=1, keepdims=True)


def binned_aupro(
    neg: np.ndarray,
    reg: np.ndarray,
    reg_weight: np.ndarray,
    fpr_limit: float = 0.3,
    frac_above: np.ndarray | None = None,
) -> float:
    """AUPRO from summed normal-pixel counts (1-D), region histograms (R, B) and region weights (R,).

    Thresholds sit at bin boundaries, from the top bin down. At each one, FPR is the fraction of
    normal pixels above it and PRO the weighted mean, over regions, of the fraction above it.
    """
    neg = neg.astype(np.float64)
    fpr = np.r_[0.0, np.cumsum(neg[::-1]) / neg.sum()]
    if frac_above is None:
        frac_above = region_frac_above(reg)
    pro = np.r_[0.0, reg_weight @ frac_above / reg_weight.sum()]
    keep = fpr <= fpr_limit
    fpr_cut, pro_cut = fpr[keep], pro[keep]
    if fpr_cut[-1] < fpr_limit:
        fpr_cut, pro_cut = np.r_[fpr_cut, fpr_limit], np.r_[pro_cut, np.interp(fpr_limit, fpr, pro)]
    return float(np.trapezoid(pro_cut, fpr_cut) / fpr_limit)


def stats_metrics(s: dict[str, np.ndarray], img_weight: np.ndarray | None = None) -> tuple[float, float]:
    """(pixel AUROC, AUPRO) of a category, with each image counted img_weight[i] times."""
    w = np.ones(len(s["neg"])) if img_weight is None else img_weight.astype(np.float64)
    neg, pos = w @ s["neg"], w @ s["pos"]
    return binned_pixel_auroc(neg, pos), binned_aupro(neg, s["reg"], w[s["reg_img"]])


def save_stats(path: Path, per_category: dict[str, dict[str, np.ndarray]], paths: dict[str, list[str]]) -> None:
    """One .npz for a run: arrays named '<category>/<field>', plus the image paths in row order."""
    arrays = {f"{c}/{k}": v for c, s in per_category.items() for k, v in s.items()}
    arrays.update({f"{c}/paths": np.array(p) for c, p in paths.items()})
    np.savez_compressed(path, **arrays)


def load_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        cat, field = key.rsplit("/", 1)
        out.setdefault(cat, {})[field] = z[key]
    return out


def run_pixel_stats(rdir: Path, dataset_name: str, paths: str, out: Path, eval_size: int | None = None) -> float:
    """Write out/pixel_stats.npz and out/pixel_stats_check.json for one run; return max |exact - binned|."""
    from data.registry import build_dataset
    from eval.cache import EVAL_SIZE, load_map, read_scores

    eval_size = eval_size or EVAL_SIZE
    rdir, out = Path(rdir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(dataset_name, paths)
    if not dataset.has_pixel_labels:
        raise ValueError(f"{dataset_name} has no pixel labels")
    index_of = {r.image_path: i for i, r in enumerate(dataset.records)}
    rows = read_scores(rdir)

    per_category, img_paths, check = {}, {}, {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        masks = np.stack([dataset.load_mask(index_of[r["path"]], eval_size) for r in cat_rows]).astype(bool)
        if not masks.any():
            continue
        maps = np.stack([load_map(rdir, r["path"]) for r in cat_rows])
        s = compute_stats(masks, maps)
        per_category[cat], img_paths[cat] = s, [r["path"] for r in cat_rows]
        exact = (pixel_auroc(masks, maps), aupro(masks, maps))
        binned = stats_metrics(s)
        check[cat] = {"exact": exact, "binned": binned}
        print(
            f"{cat:16s} pixel AUROC exact {100 * exact[0]:6.2f} binned {100 * binned[0]:6.2f}   "
            f"AUPRO exact {100 * exact[1]:6.2f} binned {100 * binned[1]:6.2f}",
            flush=True,
        )
    save_stats(out / "pixel_stats.npz", per_category, img_paths)
    worst = max(abs(v["exact"][k] - v["binned"][k]) for v in check.values() for k in (0, 1))
    (out / "pixel_stats_check.json").write_text(json.dumps({"max_abs_diff": worst, "per_category": check}, indent=2))
    print(f"max |exact - binned| = {100 * worst:.3f} points", flush=True)
    return worst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--paths", default="kaggle")
    p.add_argument("--out", required=True, help="folder for pixel_stats.npz and pixel_stats_check.json")
    p.add_argument("--eval-size", type=int, default=None)
    a = p.parse_args()
    run_pixel_stats(Path(a.run), a.dataset, a.paths, Path(a.out), a.eval_size)


if __name__ == "__main__":
    main()
