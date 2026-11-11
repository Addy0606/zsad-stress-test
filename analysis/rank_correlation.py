"""Rank correlation of image scores between detectors (spec Phase 5: complementarity).

For each dataset and each pair of the four detectors: Spearman's rho between their image scores, computed within
each category (scores are compared per category everywhere in this project) and averaged over categories.
95% CIs: percentile bootstrap, 1,000 resamples of images stratified by category and label (eval/bootstrap.py).
Low rho means two detectors order the images differently, which is what lets them miss different images.

Reads the per-image scores in outputs/kaggle/zsad-p5-mapscores/ (aligned across detectors by image path);
writes analysis/results/rank_correlation.csv and analysis/figures/rank_correlation.png.

    python analysis/rank_correlation.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.figures import DATASETS, INK, METHODS, MUTED, SURFACE, style  # noqa: E402
from eval.bootstrap import percentile_ci, resample_counts  # noqa: E402
from fusion.image_fusion import DETECTORS, Data, own_score_cols  # noqa: E402

N_RESAMPLES = 1000
SEED = 0
ORDER = ("mvtec", "visa", "mpdd", "bmad_brain", "headct")
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")  # sequential blue 100-700


def category_rho(a: np.ndarray, b: np.ndarray, cats: np.ndarray, counts: np.ndarray | None = None) -> float:
    """Spearman rho between scores a and b within each category (images repeated `counts` times), averaged."""
    vals = []
    for c in np.unique(cats):
        idx = np.flatnonzero(cats == c)
        if counts is not None:
            idx = np.repeat(idx, counts[idx])
        vals.append(spearmanr(a[idx], b[idx]).statistic)
    return float(np.mean(vals))


def dataset_rows(dataset: str) -> list[dict]:
    d = Data(dataset)
    scores = {m: d.X[:, c] for m, c in zip(DETECTORS, own_score_cols(), strict=True)}
    pairs = list(itertools.combinations(DETECTORS, 2))
    rng = np.random.default_rng(SEED)
    reps = {p: [] for p in pairs}
    for _ in range(N_RESAMPLES):
        counts = np.zeros(len(d.labels), dtype=int)
        for c in np.unique(d.cats):
            sel = np.flatnonzero(d.cats == c)
            counts[sel] = resample_counts(d.labels[sel], rng)
        for a, b in pairs:
            reps[(a, b)].append(category_rho(scores[a], scores[b], d.cats, counts))
    rows = []
    for a, b in pairs:
        lo, hi = percentile_ci(np.array(reps[(a, b)]))
        rho = category_rho(scores[a], scores[b], d.cats)
        rows.append({"dataset": dataset, "method_a": a, "method_b": b, "rho": rho, "ci_low": lo, "ci_high": hi})
    return rows


def colour(v: float) -> str:
    return RAMP[int(np.clip(v, 0, 0.999) * len(RAMP))]


def figure(rows: list[dict], out: Path) -> None:
    style()
    by = {(r["dataset"], r["method_a"], r["method_b"]): r for r in rows}
    labels = [METHODS[m][0].replace("AnomalyVFM ", "VFM ") for m in DETECTORS]
    fig, axes = plt.subplots(1, len(ORDER), figsize=(13, 3.3))
    for ax, ds in zip(axes, ORDER, strict=True):
        n = len(DETECTORS)
        for i, j in itertools.product(range(n), range(n)):
            if i == j:
                continue
            a, b = sorted((DETECTORS[i], DETECTORS[j]), key=DETECTORS.index)
            v = float(by[(ds, a, b)]["rho"])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=colour(v), ec=SURFACE, lw=2))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v >= 4 / 7 else INK)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n), labels, rotation=40, ha="right")
        ax.set_yticks(range(n), labels if ds == ORDER[0] else [])
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(DATASETS[ds], loc="left", fontsize=10)
    fig.suptitle("Spearman rank correlation of image scores (mean over categories)", x=0.01, ha="left", color=INK)
    fig.text(
        0.01, 0.01, "1 = same ordering of images; low = the detectors rank images differently.", color=MUTED, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = []
    for ds in ORDER:
        rows += dataset_rows(ds)
        print("done", ds, flush=True)
    out = REPO / "analysis" / "results" / "rank_correlation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    figure(rows, REPO / "analysis" / "figures" / "rank_correlation.png")
    for r in rows:
        print(
            f"{r['dataset']:10s} {r['method_a']:18s} {r['method_b']:18s} "
            f"{r['rho']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
