"""Phase 6 results: bootstrap CIs for the pixel-fusion heads and the fused-map image score, then the ablation tables.

Inputs: results/fusion_image.csv (fusion/image_fusion.py) and the pixel-fusion job's outputs, fetched into
outputs/kaggle/zsad-p6-pixelfusion/pixel_fusion/<test>/<head>/ (metrics.json, pixel_stats.npz, image_scores.csv).
Outputs: results/fusion_pixel.csv, results/fusion_image_from_maps.csv, results/fusion.md

Pixel heads: point values are the exact metrics from metrics.json; CIs and the paired difference to best_single
come from the per-image histograms (eval/pixel_stats.py) with the SAME resampled images for every head.
Fused-map image score: image AUROC of "mean of the top 1% of the fused map", paired against the best single
image method (AnomalyVFM RADIO's own score, which the training data selects in every split).

    python fusion/report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.bootstrap import bootstrap_metrics, percentile_ci, resample_counts  # noqa: E402
from eval.pixel_stats import load_stats  # noqa: E402
from fusion.image_fusion import DETECTORS, FEATURES, Data, category_auroc, missed_at_5pct  # noqa: E402

PIX = REPO / "outputs" / "kaggle" / "zsad-p6-pixelfusion" / "pixel_fusion"
RES = REPO / "results"
PIXEL_HEADS = ("best_single", "mean", "logreg")
N_RESAMPLES = 1000
SEED = 0
NAMES = {"mvtec": "MVTec AD", "visa": "VisA", "bmad_brain": "BMAD Brain MRI", "mpdd": "MPDD", "headct": "Head CT"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def pixel_rows(test: str) -> list[dict]:
    """Exact pixel metrics + bootstrap CIs and paired differences to best_single, for one held-out dataset."""
    reps, exact, chosen = {}, {}, {}
    for h in PIXEL_HEADS:
        hdir = PIX / test / h
        scores = read_csv(hdir / "image_scores.csv")
        rows = [
            {"category": r["category"], "label": int(r["label"]), "score": float(r["score"]), "path": r["image"]}
            for r in scores
        ]
        rows.sort(key=lambda r: r["path"])  # same order for every head -> identical resamples
        reps[h] = bootstrap_metrics(rows, load_stats(hdir / "pixel_stats.npz"), N_RESAMPLES, seed=SEED)
        m = json.loads((hdir / "metrics.json").read_text(encoding="utf-8"))
        exact[h], chosen[h] = m["mean"], {"best_single": m["best_single"], "logreg_C": m["logreg_C"]}
    out = []
    for h in PIXEL_HEADS:
        for metric in ("pixel_auroc", "aupro"):
            lo, hi = percentile_ci(reps[h][metric])
            dlo, dhi = percentile_ci(reps[h][metric] - reps["best_single"][metric])
            out.append(
                {
                    "test": test,
                    "head": h,
                    "metric": metric,
                    "value": exact[h][metric],
                    "ci_low": lo,
                    "ci_high": hi,
                    "diff_vs_best_single": exact[h][metric] - exact["best_single"][metric],
                    "diff_ci_low": dlo,
                    "diff_ci_high": dhi,
                    "best_single_is": chosen[h]["best_single"],
                    "logreg_C": chosen[h]["logreg_C"],
                }
            )
    return out


def fused_map_image_rows(test: str) -> list[dict]:
    """Image AUROC / misses of the fused-map image score vs RADIO's own score, paired."""
    d = Data(test)
    radio = d.X[:, DETECTORS.index("anomalyvfm_radio") * len(FEATURES)]
    candidates = {"radio_own_score": radio}
    for h in ("mean", "logreg"):
        by_key = {r["image"]: float(r["score"]) for r in read_csv(PIX / test / h / "image_scores.csv")}
        candidates[f"pixel_{h}_top1"] = np.array([by_key[k] for k in d.keys])
    rng = np.random.default_rng(SEED)
    reps = {k: [] for k in candidates}
    miss_reps = {k: [] for k in candidates}
    misses = {k: missed_at_5pct(v, d) for k, v in candidates.items()}
    anom = d.labels == 1
    for _ in range(N_RESAMPLES):
        counts = np.zeros(len(d.labels), dtype=int)
        for c in np.unique(d.cats):
            sel = np.flatnonzero(d.cats == c)
            counts[sel] = resample_counts(d.labels[sel], rng)
        for k, v in candidates.items():
            reps[k].append(category_auroc(v, d, counts))
            miss_reps[k].append(counts[anom] @ misses[k][anom] / counts[anom].sum())
    out = []
    base = np.array(reps["radio_own_score"])
    for k, v in candidates.items():
        auc = category_auroc(v, d)
        lo, hi = percentile_ci(np.array(reps[k]))
        dlo, dhi = percentile_ci(np.array(reps[k]) - base)
        mlo, mhi = percentile_ci(np.array(miss_reps[k]))
        out.append(
            {
                "test": test,
                "score": k,
                "image_auroc": auc,
                "ci_low": lo,
                "ci_high": hi,
                "diff_vs_radio": auc - category_auroc(radio, d),
                "diff_ci_low": dlo,
                "diff_ci_high": dhi,
                "miss_at_5pct_fpr": float(misses[k][anom].mean()),
                "miss_ci_low": mlo,
                "miss_ci_high": mhi,
            }
        )
    return out


def pct(v: str | float) -> str:
    return f"{100 * float(v):.1f}"


def ci(r: dict, lo: str = "ci_low", hi: str = "ci_high") -> str:
    return f"[{pct(r[lo])}, {pct(r[hi])}]"


def diff(r: dict, key: str, lo: str = "diff_ci_low", hi: str = "diff_ci_high") -> str:
    return f"{100 * float(r[key]):+.1f} [{100 * float(r[lo]):+.1f}, {100 * float(r[hi]):+.1f}]"


def oracle_miss() -> dict[str, dict]:
    return {
        r["dataset"]: r
        for r in read_csv(REPO / "analysis" / "results" / "complementarity.csv")
        if r["stat"] == "miss:oracle_upper_bound"
    }


def write_markdown(img: list[dict], pix: list[dict], fmap: list[dict]) -> None:
    order = ["mvtec", "visa", "bmad_brain", "mpdd", "headct"]
    heads = [
        ("best_single", "Best single (chosen on training data)"),
        ("mean_own", "Mean of own scores"),
        ("mean_all", "Mean of all 16 features"),
        ("logreg", "Logistic regression"),
        ("gbt", "Boosted trees"),
    ]
    oracle = oracle_miss()
    L = [
        "# Fusion results (Phase 6)",
        "",
        "Generated by `python fusion/report.py`; do not edit by hand. Every head is trained without the dataset",
        "in the column (leave-one-dataset-out over MVTec AD, VisA and BMAD Brain; MPDD and Head CT are never used",
        "for training).",
        "Values in %, with 95% bootstrap CIs; differences are paired (same resampled images).",
        "",
        "## Image AUROC",
        "",
        "| Head | " + " | ".join(NAMES[t] for t in order) + " |",
        "|---|" + "---|" * len(order),
    ]
    by = {(r["test"], r["head"]): r for r in img}
    for h, label in heads:
        cells = []
        for t in order:
            r = by[(t, h)]
            cells.append(
                f"{pct(r['image_auroc'])} {ci(r)}"
                if h == "best_single"
                else f"{pct(r['image_auroc'])} ({diff(r, 'diff_vs_best_single')})"
            )
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    fm = {(r["test"], r["score"]): r for r in fmap}
    for h, label in (
        ("pixel_mean_top1", "Fused map (mean), top-1% score"),
        ("pixel_logreg_top1", "Fused map (logreg), top-1% score"),
    ):
        L.append(
            f"| {label} | "
            + " | ".join(f"{pct(fm[(t, h)]['image_auroc'])} ({diff(fm[(t, h)], 'diff_vs_radio')})" for t in order)
            + " |"
        )
    L += [
        "",
        "Brackets after a head are its difference to the best single method. The best single method chosen on the",
        "training data is AnomalyVFM RADIO in every split, which is also the best single method on every test set in",
        "hindsight, so selection itself costs nothing here.",
        "",
        "## Anomalous images missed at 5% image FPR",
        "",
        "| Head | " + " | ".join(NAMES[t] for t in order) + " |",
        "|---|" + "---|" * len(order),
    ]
    for h, label in heads:
        L.append(
            f"| {label} | "
            + " | ".join(
                f"{pct(by[(t, h)]['miss_at_5pct_fpr'])} {ci(by[(t, h)], 'miss_ci_low', 'miss_ci_high')}" for t in order
            )
            + " |"
        )
    L.append(
        "| Fused map (logreg), top-1% score | "
        + " | ".join(
            f"{pct(fm[(t, 'pixel_logreg_top1')]['miss_at_5pct_fpr'])} "
            f"{ci(fm[(t, 'pixel_logreg_top1')], 'miss_ci_low', 'miss_ci_high')}"
            for t in order
        )
        + " |"
    )
    L.append(
        "| *Oracle upper bound (Phase 5, uses labels)* | "
        + " | ".join(f"{pct(oracle[t]['value'])}" for t in order)
        + " |"
    )
    pb = {(r["test"], r["head"], r["metric"]): r for r in pix}
    ptests = [t for t in order if t != "headct"]
    for metric, title in (("pixel_auroc", "Pixel AUROC"), ("aupro", "AUPRO (FPR ≤ 0.3)")):
        L += [
            "",
            f"## {title} (pixel-level fusion)",
            "",
            "| Head | " + " | ".join(NAMES[t] for t in ptests) + " |",
            "|---|" + "---|" * len(ptests),
        ]
        for h, label in (
            ("best_single", "Best single map (chosen by training AUPRO)"),
            ("mean", "Mean of normalised maps"),
            ("logreg", "Logistic regression on maps"),
        ):
            cells = []
            for t in ptests:
                r = pb[(t, h, metric)]
                cells.append(
                    f"{pct(r['value'])} {ci(r)}"
                    + (f" — {r['best_single_is']}" if h == "best_single" else f" ({diff(r, 'diff_vs_best_single')})")
                )
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    (RES / "fusion.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    img = read_csv(RES / "fusion_image.csv")
    pix, fmap = [], []
    for t in ("mvtec", "visa", "bmad_brain", "mpdd"):
        pix += pixel_rows(t)
        print("pixel", t, flush=True)
    for t in ("mvtec", "visa", "bmad_brain", "mpdd", "headct"):
        fmap += fused_map_image_rows(t)
        print("fused-map image score", t, flush=True)
    write_csv(RES / "fusion_pixel.csv", pix)
    write_csv(RES / "fusion_image_from_maps.csv", fmap)
    write_markdown(img, pix, fmap)
    print("wrote", RES / "fusion.md")


if __name__ == "__main__":
    main()
