"""Figures for analysis/FINDINGS.md, drawn only from the CSVs in analysis/results/.

    python analysis/figures.py

Style: one fixed colour per method (validated categorical palette, same everywhere), thin marks,
95% CIs as whiskers, recessive grid, a legend on every multi-series figure. Values are also in the
CSVs / FINDINGS tables, so no figure relies on colour alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "analysis" / "results"
FIG = REPO / "analysis" / "figures"

METHODS = {  # fixed order and colour per method (categorical slots 1-4)
    "anomalyclip": ("AnomalyCLIP", "#2a78d6"),
    "visualad": ("VisualAD", "#eb6834"),
    "anomalyvfm_dinov2": ("AnomalyVFM (DINOv2)", "#1baf7a"),
    "anomalyvfm_radio": ("AnomalyVFM (RADIO)", "#eda100"),
}
DATASETS = {"mvtec": "MVTec AD", "visa": "VisA", "mpdd": "MPDD", "bmad_brain": "BMAD Brain MRI", "headct": "Head CT"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9,
        }
    )


def read(name: str) -> list[dict]:
    with open(RES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dot_ci(ax, x: float, row: dict, color: str, key: str = "value", lo: str = "ci_low", hi: str = "ci_high") -> None:
    v, a, b = 100 * float(row[key]), 100 * float(row[lo]), 100 * float(row[hi])
    ax.plot([x, x], [a, b], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(x, v, "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=3)


def grouped(ax, groups: list[str], rows_by: dict, key: str, lo: str, hi: str, width: float = 0.62) -> None:
    n = len(METHODS)
    for gi, g in enumerate(groups):
        for mi, m in enumerate(METHODS):
            r = rows_by.get((g, m))
            if r is not None:
                dot_ci(ax, gi - width / 2 + width * mi / (n - 1), r, METHODS[m][1], key, lo, hi)
    ax.set_xticks(range(len(groups)), [DATASETS.get(g, g) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)


def legend(fig, ncol: int = 4) -> None:
    handles = [
        plt.Line2D([], [], marker="o", ls="-", lw=2, ms=7, color=c, mec=SURFACE, mew=2) for _, c in METHODS.values()
    ]
    fig.legend(handles, [n for n, _ in METHODS.values()], loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.99))


def fig_wrong_place() -> None:
    rows = {(r["dataset"], r["method"]): r for r in read("peak_in_mask.csv")}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    grouped(ax, ["mvtec", "visa", "mpdd", "bmad_brain"], rows, "wrong_place_among_flagged", "ci_low", "ci_high")
    ax.set_ylabel("Flagged anomalous images whose\nmap peak is off the defect (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Right image, wrong place", loc="left", pad=28)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "wrong_place.png", dpi=180)
    plt.close(fig)


def fig_size() -> None:
    rows = read("region_found_by_size.csv")
    bins = list(dict.fromkeys(r["bin"] for r in rows))
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, d in zip(axes, ["mvtec", "visa", "mpdd", "bmad_brain"], strict=True):
        for mi, m in enumerate(METHODS):
            sel = [r for r in rows if r["dataset"] == d and r["method"] == m]
            xs = [bins.index(r["bin"]) + (mi - 1.5) * 0.08 for r in sel]
            ys = [100 * float(r["found"]) for r in sel]
            ax.plot(xs, ys, color=METHODS[m][1], lw=2, zorder=2)
            for x, r in zip(xs, sel, strict=True):
                dot_ci(ax, x, r, METHODS[m][1], "found")
        ax.set_title(DATASETS[d], loc="left")
        ax.set_xticks(range(len(bins)), [b.split(" ")[0] for b in bins])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Defect regions found at 1% pixel FPR (%)")
    fig.supxlabel(
        "Defect region size (tiny < 100 px, small < 1k, medium < 5k, large ≥ 5k, at 256×256)", fontsize=9, color=MUTED
    )
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "found_by_size.png", dpi=180)
    plt.close(fig)


def fig_domain() -> None:
    rows = read("domain_drops.csv")
    targets = list(dict.fromkeys(r["target"] for r in rows))
    by = {(r["target"], r["method"]): r for r in rows if r["method"] in METHODS}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    grouped(ax, targets, by, "drop", "ci_low", "ci_high", width=0.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Image AUROC drop from\nMVTec AD + VisA (points)")
    ax.set_title("Cost of domain shift", loc="left", pad=28)
    legend(fig, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(FIG / "domain_drop.png", dpi=180)
    plt.close(fig)


def fig_complementarity() -> None:
    rows = {(r["dataset"], r["stat"]): r for r in read("complementarity.csv")}
    datasets = ["mvtec", "visa", "mpdd", "bmad_brain", "headct"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.72
    for gi, d in enumerate(datasets):
        for mi, m in enumerate(METHODS):
            dot_ci(ax, gi - width / 2 + width * mi / 5, rows[(d, f"miss:{m}")], METHODS[m][1])
        for k, (stat, marker, fill) in enumerate(
            [("miss:or_rule_5pct_total", "s", SURFACE), ("miss:oracle_upper_bound", "D", INK)]
        ):
            x = gi - width / 2 + width * (4 + k) / 5
            r = rows[(d, stat)]
            v, a, b = (100 * float(r[c]) for c in ("value", "ci_low", "ci_high"))
            ax.plot([x, x], [a, b], color=INK, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(x, v, marker, ms=6.5, color=INK, mfc=fill, mew=1.5, zorder=3)
    ax.set_xticks(range(len(datasets)), [DATASETS[d] for d in datasets])
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Anomalous images missed at 5% FPR (%)")
    ax.set_title("Different methods miss different images", loc="left", pad=40)
    handles = [plt.Line2D([], [], marker="o", lw=2, ms=7, color=c, mec=SURFACE, mew=2) for _, c in METHODS.values()]
    handles += [
        plt.Line2D([], [], marker="s", lw=2, ms=6.5, color=INK, mfc=SURFACE, mew=1.5),
        plt.Line2D([], [], marker="D", lw=2, ms=6.5, color=INK, mfc=INK),
    ]
    labels = [n for n, _ in METHODS.values()] + ["OR rule, 5% total FPR", "Oracle (upper bound, uses labels)"]
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "complementarity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_wrong_place, fig_size, fig_domain, fig_complementarity):
        try:
            f()
            print("ok  ", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, "(missing", Path(e.filename).name + ")")


if __name__ == "__main__":
    main()
