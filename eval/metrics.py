"""Evaluation metrics: image AUROC, image F1-max, pixel AUROC, AUPRO (FPR <= 0.3).

All functions take plain numpy arrays and are computed per category; dataset numbers are the
mean over categories (the convention used by MVTec AD, VisA and the papers we compare against).

    image_auroc(labels, scores)        labels: (N,) {0,1}; scores: (N,)
    image_f1_max(labels, scores)
    pixel_auroc(masks, maps)           masks: (N, H, W) {0,1}; maps: (N, H, W) float
    aupro(masks, maps, fpr_limit=0.3)

AUPRO, in words: for a threshold t, the PRO is the average, over every connected defect region
in the ground truth, of the fraction of that region's pixels scoring >= t. So a tiny defect counts
as much as a huge one. The FPR is the fraction of normal pixels scoring >= t. AUPRO is the area
under the PRO-vs-FPR curve from FPR 0 to 0.3, divided by 0.3 so a perfect detector scores 1.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score

EIGHT_CONNECTED = np.ones((3, 3), dtype=int)  # as in the official MVTec AD evaluation code


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def image_f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 over all thresholds."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """AUROC over all pixels of all images. Needs both normal and defect pixels."""
    return float(roc_auc_score(masks.ravel().astype(np.uint8), maps.ravel()))


def pro_curve(masks: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact PRO-vs-FPR curve, one point per distinct score value.

    Every defect pixel gets weight 1 / (size of its region * number of regions), every normal
    pixel gets weight 1 / (number of normal pixels). Sorting pixels by score (highest first)
    and taking cumulative sums of these weights gives PRO and FPR for every threshold at once.
    """
    masks = masks.astype(bool)
    pro_weight = np.zeros(masks.shape, dtype=np.float64)
    n_regions = 0
    region_sizes = []
    for i in range(len(masks)):
        labelled, n = ndimage.label(masks[i], structure=EIGHT_CONNECTED)
        for r in range(1, n + 1):
            region = labelled == r
            pro_weight[i][region] = 1.0 / region.sum()
            region_sizes.append(region.sum())
        n_regions += n
    if n_regions == 0:
        raise ValueError("AUPRO needs at least one defect region")
    pro_weight /= n_regions

    n_normal = (~masks).sum()
    if n_normal == 0:
        raise ValueError("AUPRO needs at least one normal pixel")
    fpr_weight = (~masks).astype(np.float64) / n_normal

    order = np.argsort(-maps.ravel(), kind="stable")
    scores_sorted = maps.ravel()[order]
    pro = np.cumsum(pro_weight.ravel()[order])
    fpr = np.cumsum(fpr_weight.ravel()[order])

    # Pixels with equal scores are all included by the same threshold: keep the last of each run.
    last_of_run = np.r_[scores_sorted[1:] != scores_sorted[:-1], True]
    pro, fpr = pro[last_of_run], fpr[last_of_run]
    return np.r_[0.0, fpr], np.r_[0.0, pro]  # curve starts at (0, 0): threshold above every score


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Normalised area under the PRO curve up to `fpr_limit`."""
    fpr, pro = pro_curve(masks, maps)
    # Cut the curve at fpr_limit, adding an interpolated point exactly at the limit.
    keep = fpr <= fpr_limit
    fpr_cut, pro_cut = fpr[keep], pro[keep]
    if fpr_cut[-1] < fpr_limit:
        pro_at_limit = np.interp(fpr_limit, fpr, pro)
        fpr_cut, pro_cut = np.r_[fpr_cut, fpr_limit], np.r_[pro_cut, pro_at_limit]
    return float(np.trapezoid(pro_cut, fpr_cut) / fpr_limit)


def category_metrics(
    labels: np.ndarray, scores: np.ndarray, masks: np.ndarray | None, maps: np.ndarray | None
) -> dict[str, float]:
    """All metrics for one category. Pixel metrics are skipped (NaN) when masks are None."""
    out = {
        "image_auroc": image_auroc(labels, scores),
        "image_f1_max": image_f1_max(labels, scores),
        "pixel_auroc": float("nan"),
        "aupro": float("nan"),
    }
    if masks is not None and maps is not None and masks.any():
        out["pixel_auroc"] = pixel_auroc(masks, maps)
        out["aupro"] = aupro(masks, maps)
    return out
