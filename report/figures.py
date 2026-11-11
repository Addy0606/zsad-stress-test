"""Figures for the README (Phase 8), drawn from committed result tables only.

    python report/figures.py

- overview.png  the pipeline: datasets -> detectors -> cached outputs -> evaluation -> analysis / fusion / demo
- fusion.png    what fusion gains over the best single method, per held-out dataset: pixel level (AUPRO, mean
                of normalised maps vs best single map) next to image level (image AUROC, logistic regression vs
                best single score). Paired 95% bootstrap CIs from results/fusion_pixel.csv and fusion_image.csv.
Style and colours: shared with analysis/figures.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.figures import DATASETS, GRID, INK, METHODS, MUTED, SURFACE, style  # noqa: E402

OUT = REPO / "report" / "figures"
MARK = "#3d3c39"  # single-series marks: neutral ink, so they don't read as one of the methods
ORDER = ("mvtec", "visa", "bmad_brain", "mpdd", "headct")


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def box(ax, x: float, y: float, w: float, h: float, title: str, lines: list[str], chips=()) -> None:
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=GRID, lw=1.2)
    )
    ax.text(x + 0.015, y + h - 0.035, title, ha="left", va="top", fontsize=9.5, color=INK, weight="bold")
    for i, line in enumerate(lines):
        yy = y + h - 0.11 - i * 0.056
        if i < len(chips):  # colour chip = the method's colour in every figure
            ax.add_patch(plt.Rectangle((x + 0.017, yy - 0.012), 0.011, 0.024, color=chips[i], lw=0))
            ax.text(x + 0.035, yy, line, ha="left", va="center", fontsize=8, color=MUTED)
        else:
            ax.text(x + 0.017, yy, line, ha="left", va="center", fontsize=8, color=MUTED)


def arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))


def fig_overview() -> None:
    fig = plt.figure(figsize=(10, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, h = 0.2, 0.56
    box(ax, 0.01, y, 0.17, h, "5 datasets", ["MVTec AD", "VisA", "MPDD (test only)", "BMAD Brain MRI", "Head CT"])
    box(
        ax,
        0.215,
        y,
        0.215,
        h,
        "4 zero-shot detectors",
        [*[METHODS[m][0] for m in METHODS], "+ B1 CLIP prompt", "+ B2 DINOv2 NN (transductive)"],
        chips=[METHODS[m][1] for m in METHODS],
    )
    box(ax, 0.465, y, 0.16, h, "Cached outputs", ["per image:", "score", "anomaly map", "(Kaggle T4 GPU)"])
    box(ax, 0.66, y, 0.15, h, "Evaluation", ["image AUROC", "pixel AUROC", "AUPRO", "F1-max", "95% bootstrap CIs"])
    box(ax, 0.845, y, 0.14, h, "Uses", ["failure analysis", "fusion, leave-one-", "dataset-out", "Gradio demo"])
    for x0, x1 in ((0.18, 0.215), (0.43, 0.465), (0.625, 0.66), (0.81, 0.845)):
        arrow(ax, x0, y + h / 2, x1, y + h / 2)
    ax.text(0.01, 0.93, "Pipeline", fontsize=11, color=INK, weight="bold", ha="left", va="center")
    ax.text(
        0.01,
        0.08,
        "Every number is computed from the cached outputs. No detector is trained on the dataset it is scored on "
        "(B2 compares unlabelled test images with each other and is reported separately).",
        fontsize=8,
        color=MUTED,
        ha="left",
    )
    fig.savefig(OUT / "overview.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_fusion() -> None:
    pix = {(r["test"], r["head"], r["metric"]): r for r in read(REPO / "results" / "fusion_pixel.csv")}
    img = {(r["test"], r["head"]): r for r in read(REPO / "results" / "fusion_image.csv")}
    panels = (
        (
            "Pixel level: mean of normalised maps\nvs best single map (AUPRO)",
            [(t, pix.get((t, "mean", "aupro"))) for t in ORDER],
        ),
        (
            "Image level: logistic regression\nvs best single score (image AUROC)",
            [(t, img.get((t, "logreg"))) for t in ORDER],
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, (title, rows) in zip(axes, panels, strict=True):
        ax.axhline(0, color=MUTED, lw=1)
        for i, (_t, r) in enumerate(rows):
            if r is None:  # Head CT has no masks: no pixel metrics
                ax.text(i, 0.3, "no masks", ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            v, lo, hi = (100 * float(r[k]) for k in ("diff_vs_best_single", "diff_ci_low", "diff_ci_high"))
            ax.plot([i, i], [lo, hi], color=MARK, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(i, v, "o", ms=7, color=MARK, mec=SURFACE, mew=2, zorder=3)
            ax.text(i + 0.12, v, f"{v:+.1f}", ha="left", va="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(ORDER)), [DATASETS[t].replace(" Brain MRI", "\nBrain MRI") for t in ORDER])
        ax.set_xlim(-0.5, len(ORDER) - 0.4)
        ax.set_title(title, loc="left", fontsize=9.5)
    axes[0].set_ylabel("difference to best single method (points)")
    fig.text(
        0.01,
        0.01,
        "Held-out datasets only; whiskers are paired 95% bootstrap CIs. MPDD and Head CT were never used for training.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fusion.png", dpi=180)
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_overview()
    fig_fusion()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
