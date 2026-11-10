"""Image-level failure analysis from per-image scores (Phase 5, RQ1 defect types, RQ2, RQ3a).

Reads the scores.csv of every run in configs/bench_sources.yaml (fetched under outputs/kaggle/) and writes
tables to analysis/results/. All CIs are 95% percentile bootstrap, 1,000 resamples, images resampled within
each category and label (same scheme as eval/bootstrap.py).

Definitions
- Operating point "5% FPR": per method and category, the image-score threshold exceeded by 5% of the
  category's normal images. An anomalous image is *missed* if its score is not above that threshold.
  Thresholds are re-estimated inside every bootstrap resample, so the CIs include threshold uncertainty.
- Oracle (upper bound, uses labels): an anomalous image counts as caught if at least one method catches
  it at its own 5% FPR. Not achievable: OR-ing several detectors also OR-s their false alarms
  (`fpr:any_at_5pct` reports how many normal images at least one method flags).
- OR rule at 5% total FPR (achievable, no labels): flag an image if any method's score is above its
  threshold, with all methods' thresholds lowered together (same per-method FPR alpha, found by bisection
  per category) until only 5% of normal images are flagged in total. A simple fusion baseline for Phase 6.
- Domain drop: image AUROC on the standard benchmarks (mean of MVTec AD and VisA) minus image AUROC on the
  target (MPDD; or the mean of BMAD Brain and Head CT). Datasets are resampled independently.

    python analysis/image_level.py
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.bootstrap import bootstrap_metrics, percentile_ci, resample_counts  # noqa: E402
from eval.cache import read_scores, relative_path, run_dataset_root  # noqa: E402
from eval.metrics import image_auroc  # noqa: E402

N_RESAMPLES = 1000
FPR = 0.05
OUT = REPO / "analysis" / "results"


def load_runs(kaggle_dir: Path) -> tuple[dict, dict]:
    """{(method, dataset): rows with an extra 'key' = path relative to the dataset root}, manifest."""
    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    runs = {}
    for method, by_dataset in manifest.runs.items():
        for dataset, src in by_dataset.items():
            rdir = kaggle_dir / src.kernel / "outputs" / src.run
            root = run_dataset_root(rdir)
            rows = read_scores(rdir)
            for r in rows:
                r["key"] = relative_path(r["path"], root)
            runs[(method, dataset)] = rows
    return runs, manifest


def aligned(runs: dict, methods: list[str], dataset: str):
    """Scores of several methods on the same images: keys, categories, labels, defect types, scores (M, N)."""
    by_key = [{r["key"]: r for r in runs[(m, dataset)]} for m in methods]
    keys = sorted(by_key[0])
    for k in by_key[1:]:
        assert set(k) == set(keys), f"methods scored different images on {dataset}"
    first = by_key[0]
    cats = np.array([first[k]["category"] for k in keys])
    labels = np.array([first[k]["label"] for k in keys])
    types = np.array([first[k]["defect_type"] for k in keys])
    scores = np.array([[b[k]["score"] for k in keys] for b in by_key])
    return keys, cats, labels, types, scores


def caught(scores: np.ndarray, cats: np.ndarray, labels: np.ndarray, weights: np.ndarray, fpr: float = FPR):
    """(M, N) bool: image detected at `fpr`, with thresholds from the (weighted) normal images of each category."""
    out = np.zeros(scores.shape, dtype=bool)
    for c in np.unique(cats):
        normal = (cats == c) & (labels == 0)
        for m in range(len(scores)):
            vals = np.repeat(scores[m, normal], weights[normal])
            t = np.quantile(vals, 1 - fpr)
            out[m, cats == c] = scores[m, cats == c] > t
    return out


def or_rule(
    scores: np.ndarray, cats: np.ndarray, labels: np.ndarray, weights: np.ndarray, fpr: float = FPR, iters: int = 25
) -> np.ndarray:
    """(N,) bool: flagged by any method, with a common per-method FPR chosen so the union flags `fpr` of normals."""
    out = np.zeros(scores.shape[1], dtype=bool)
    for c in np.unique(cats):
        idx, normal = cats == c, (cats == c) & (labels == 0)
        nv = np.repeat(scores[:, normal], weights[normal], axis=1)  # (M, weighted normals)

        lo, hi = 0.0, fpr
        for _ in range(iters):
            mid = (lo + hi) / 2
            if (nv > np.quantile(nv, 1 - mid, axis=1)[:, None]).any(0).mean() > fpr:
                hi = mid
            else:
                lo = mid
        out[idx] = (scores[:, idx] > np.quantile(nv, 1 - lo, axis=1)[:, None]).any(0)
    return out


def complementarity(runs: dict, methods: list[str], dataset: str, seed: int = 0) -> list[dict]:
    """Miss rates at 5% FPR of each method, the oracle, and pairwise overlap, with bootstrap CIs."""
    _, cats, labels, _, scores = aligned(runs, methods, dataset)
    anom = labels == 1

    def stats(w: np.ndarray) -> dict[str, float]:
        hit = caught(scores, cats, labels, w)
        wa = w[anom].astype(float)
        miss = ~hit[:, anom]
        res = {f"miss:{m}": float(wa @ miss[i] / wa.sum()) for i, m in enumerate(methods)}
        res["miss:oracle_upper_bound"] = float(wa @ miss.all(0) / wa.sum())
        wn = w[~anom].astype(float)
        res["fpr:any_at_5pct"] = float(wn @ hit[:, ~anom].any(0) / wn.sum())
        res["miss:or_rule_5pct_total"] = float(wa @ ~or_rule(scores, cats, labels, w)[anom] / wa.sum())
        for (i, a), (j, b) in combinations(enumerate(methods), 2):
            both, either = miss[i] & miss[j], miss[i] | miss[j]
            res[f"jaccard:{a}|{b}"] = float(wa @ both / max(wa @ either, 1e-12))
        return res

    point = stats(np.ones(len(labels), dtype=int))
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(N_RESAMPLES):
        w = np.zeros(len(labels), dtype=int)
        for c in np.unique(cats):
            sel = np.flatnonzero(cats == c)
            w[sel] = resample_counts(labels[sel], rng)
        for k, v in stats(w).items():
            reps[k].append(v)
    return [
        {
            "dataset": dataset,
            "stat": k,
            "value": v,
            **dict(zip(("ci_low", "ci_high"), percentile_ci(np.array(reps[k])), strict=True)),
        }
        for k, v in point.items()
    ]


def defect_type_auroc(rows: list[dict], seed: int = 0) -> list[dict]:
    """Image AUROC of each defect type against the normal images of its category, with bootstrap CI."""
    rng = np.random.default_rng(seed)
    out = []
    for cat in sorted({r["category"] for r in rows}):
        normal = np.array([r["score"] for r in rows if r["category"] == cat and r["label"] == 0])
        for dt in sorted({r["defect_type"] for r in rows if r["category"] == cat and r["label"] == 1}):
            anom = np.array([r["score"] for r in rows if r["category"] == cat and r["defect_type"] == dt])
            labels = np.r_[np.zeros(len(normal)), np.ones(len(anom))]
            scores = np.r_[normal, anom]
            reps = []
            for _ in range(N_RESAMPLES):
                idx = np.repeat(np.arange(len(labels)), resample_counts(labels.astype(int), rng))
                reps.append(image_auroc(labels[idx], scores[idx]))
            lo, hi = percentile_ci(np.array(reps))
            out.append(
                {
                    "category": cat,
                    "defect_type": dt,
                    "n_anomalous": len(anom),
                    "auroc": image_auroc(labels, scores),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return out


def domain_drops(runs: dict, methods: list[str]) -> list[dict]:
    """Image-AUROC drop from the standard benchmarks to MPDD and to medical, per method, with CIs."""
    out = []
    for m in methods:
        reps = {
            d: bootstrap_metrics(runs[(m, d)], None, N_RESAMPLES, seed=k)["image_auroc"]
            for k, d in enumerate(["mvtec", "visa", "mpdd", "bmad_brain", "headct"])
        }
        point = {d: float(np.mean(_category_aurocs(runs[(m, d)]))) for d in reps}
        std_rep, std_pt = (reps["mvtec"] + reps["visa"]) / 2, (point["mvtec"] + point["visa"]) / 2
        targets = {
            "MPDD": (reps["mpdd"], point["mpdd"]),
            "Medical (Brain, Head CT)": (
                (reps["bmad_brain"] + reps["headct"]) / 2,
                (point["bmad_brain"] + point["headct"]) / 2,
            ),
        }
        for name, (rep, pt) in targets.items():
            lo, hi = percentile_ci(std_rep - rep)
            out.append(
                {
                    "method": m,
                    "target": name,
                    "standard": std_pt,
                    "target_auroc": pt,
                    "drop": std_pt - pt,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return out


def _category_aurocs(rows: list[dict]) -> list[float]:
    out = []
    for cat in sorted({r["category"] for r in rows}):
        lab = np.array([r["label"] for r in rows if r["category"] == cat])
        sc = np.array([r["score"] for r in rows if r["category"] == cat])
        out.append(image_auroc(lab, sc))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def main() -> None:
    runs, manifest = load_runs(REPO / "outputs" / "kaggle")
    single = [m for m in manifest.methods if not manifest.methods[m].get("transductive", False)]
    main_methods = [m for m in single if m != "clip_prompt"]  # the four detectors, without the B1 baseline

    print("domain drops ...", flush=True)
    write_csv(OUT / "domain_drops.csv", domain_drops(runs, single))

    print("complementarity ...", flush=True)
    rows = []
    for d in manifest.datasets:
        rows += complementarity(runs, main_methods, d)
    write_csv(OUT / "complementarity.csv", rows)

    print("defect types ...", flush=True)
    rows = []
    for m in single:
        for d in ("mvtec", "visa", "mpdd"):
            rows += [{"method": m, "dataset": d, **r} for r in defect_type_auroc(runs[(m, d)])]
    write_csv(OUT / "defect_type_auroc.csv", rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
