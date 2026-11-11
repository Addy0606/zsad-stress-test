"""Image-level fusion heads with leave-one-dataset-out evaluation (Phase 6).

Features (per image, per detector): the detector's own image score, and three scores read off its anomaly map
(max, mean of top 1%, mean of top 5%), from the Phase 5 table outputs/kaggle/zsad-p5-mapscores/.
4 detectors x 4 = 16 features. B1 and B2 are not inputs (B1 is a baseline; B2 is transductive).

Heads (all fitted on the training datasets of a split only, see fusion/splits.py):
- best_single   the detector whose own score has the highest mean image AUROC on the training datasets
- mean_own      mean of the 4 detectors' own scores after quantile normalisation (no learning)
- mean_all      mean of all 16 normalised features (no learning)
- logreg        logistic regression on the 16 normalised features; C chosen by inner leave-one-dataset-out
- gbt           small gradient-boosted trees on the 16 normalised features; depth/iterations chosen the same way
Reference (not a head): oracle_select = the single detector that is best ON THE TEST DATASET (hindsight).

Training weights: each training dataset counts equally, and normal/anomalous are balanced within it, so the
3,715 Brain MRI images don't drown out the others.

Evaluation per held-out dataset: image AUROC (mean over categories) with 95% bootstrap CIs, and the paired
difference to best_single (same resampled images for every head). Also the share of anomalous images missed at
5% image FPR (thresholds fixed on the full test data, as in Phase 5).

    python fusion/image_fusion.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.image_level import fpr_threshold  # noqa: E402
from eval.bootstrap import percentile_ci, resample_counts  # noqa: E402
from eval.metrics import image_auroc  # noqa: E402
from fusion.normalize import QuantileNormalizer  # noqa: E402
from fusion.splits import Split, outer_splits  # noqa: E402

DETECTORS = ("anomalyclip", "visualad", "anomalyvfm_dinov2", "anomalyvfm_radio")
FEATURES = ("score", "map_max", "map_top1", "map_top5")
MAP_SCORES = REPO / "outputs" / "kaggle" / "zsad-p5-mapscores" / "map_scores"
OUT = REPO / "results"
N_RESAMPLES = 1000
SEED = 0
LOGREG_C = (0.01, 0.1, 1.0, 10.0)
GBT_GRID = tuple((d, n) for d in (2, 3) for n in (50, 150))  # (max_depth, max_iter)
HEADS = ("best_single", "mean_own", "mean_all", "logreg", "gbt", "oracle_select")


class Data:
    """Aligned features of one dataset: keys, categories, labels, X (N, 16)."""

    def __init__(self, dataset: str, root: Path = MAP_SCORES) -> None:
        tables = []
        for m in DETECTORS:
            with open(root / m / f"{dataset}.csv", newline="", encoding="utf-8") as f:
                tables.append({r["image"]: r for r in csv.DictReader(f)})
        keys = sorted(tables[0])
        for t in tables[1:]:
            assert set(t) == set(keys), f"detectors scored different images on {dataset}"
        self.name = dataset
        self.keys = keys
        self.cats = np.array([tables[0][k]["category"] for k in keys])
        self.labels = np.array([int(tables[0][k]["label"]) for k in keys])
        self.X = np.array([[float(t[k][f]) for t in tables for f in FEATURES] for k in keys])
        self.columns = [f"{m}:{f}" for m in DETECTORS for f in FEATURES]


def own_score_cols() -> list[int]:
    return [i * len(FEATURES) for i in range(len(DETECTORS))]


def category_auroc(scores: np.ndarray, d: Data, counts: np.ndarray | None = None) -> float:
    """Image AUROC averaged over categories, optionally with images repeated `counts` times."""
    vals = []
    for c in np.unique(d.cats):
        idx = np.flatnonzero(d.cats == c)
        if counts is not None:
            idx = np.repeat(idx, counts[idx])
        vals.append(image_auroc(d.labels[idx], scores[idx]))
    return float(np.mean(vals))


def training_set(datasets: list[Data]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked X, y and sample weights (each dataset equal total weight, classes balanced within it)."""
    xs, ys, ws = [], [], []
    for d in datasets:
        w = np.where(d.labels == 1, 0.5 / max(d.labels.sum(), 1), 0.5 / max((1 - d.labels).sum(), 1))
        xs.append(d.X)
        ys.append(d.labels)
        ws.append(w)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)


def balanced_sample(datasets: list[Data], rng: np.random.Generator, n_per_dataset: int = 1500) -> np.ndarray:
    """Rows drawn equally from each training dataset, to fit the normaliser without one dataset dominating."""
    return np.vstack([d.X[rng.choice(len(d.X), min(n_per_dataset, len(d.X)), replace=False)] for d in datasets])


def fit_head(head: str, train: list[Data], param=None):
    """Return a function X -> scores, fitted on `train` only."""
    rng = np.random.default_rng(SEED)
    norm = QuantileNormalizer().fit(balanced_sample(train, rng))
    if head == "best_single":
        aucs = [np.mean([category_auroc(d.X[:, c], d) for d in train]) for c in own_score_cols()]
        col = own_score_cols()[int(np.argmax(aucs))]
        return (lambda X, col=col: X[:, col]), DETECTORS[col // len(FEATURES)]
    if head == "mean_own":
        return (lambda X: norm.transform(X)[:, own_score_cols()].mean(1)), None
    if head == "mean_all":
        return (lambda X: norm.transform(X).mean(1)), None
    X, y, w = training_set(train)
    Xn = norm.transform(X)
    if head == "logreg":
        model = LogisticRegression(C=param, max_iter=2000).fit(Xn, y, sample_weight=w)
        return (lambda X: model.decision_function(norm.transform(X))), param
    if head == "gbt":
        depth, iters = param
        model = HistGradientBoostingClassifier(max_depth=depth, max_iter=iters, learning_rate=0.1, random_state=SEED)
        model.fit(Xn, y, sample_weight=w)
        return (lambda X: model.predict_proba(norm.transform(X))[:, 1]), param
    raise ValueError(head)


def choose_param(head: str, split: Split, data: dict[str, Data]):
    """Hyperparameter with the best mean validation AUROC over the inner leave-one-dataset-out splits."""
    grid = {"logreg": LOGREG_C, "gbt": GBT_GRID}.get(head)
    if grid is None:
        return None
    best, best_score = None, -1.0
    for p in grid:
        score = np.mean(
            [category_auroc(fit_head(head, [data[t] for t in tr], p)[0](data[v].X), data[v]) for tr, v in split.inner()]
        )
        if score > best_score:
            best, best_score = p, score
    return best


def missed_at_5pct(scores: np.ndarray, d: Data) -> np.ndarray:
    """(N,) bool, True for anomalous images not above the category's 5%-FPR threshold (fixed on full data)."""
    miss = np.zeros(len(scores), dtype=bool)
    for c in np.unique(d.cats):
        idx = d.cats == c
        t = fpr_threshold(scores[idx & (d.labels == 0)], 0.05)
        miss[idx & (d.labels == 1)] = scores[idx & (d.labels == 1)] <= t
    return miss


def evaluate(split: Split, data: dict[str, Data]) -> tuple[list[dict], dict]:
    test = data[split.test]
    train = [data[t] for t in split.train]
    scores, chosen = {}, {}
    for head in HEADS:
        if head == "oracle_select":
            aucs = [category_auroc(test.X[:, c], test) for c in own_score_cols()]
            col = own_score_cols()[int(np.argmax(aucs))]
            scores[head], chosen[head] = test.X[:, col], DETECTORS[col // len(FEATURES)]
            continue
        param = choose_param(head, split, data)
        fn, info = fit_head(head, train, param)
        scores[head], chosen[head] = fn(test.X), info
    rng = np.random.default_rng(SEED)
    reps = {h: [] for h in HEADS}
    miss_reps = {h: [] for h in HEADS}
    misses = {h: missed_at_5pct(scores[h], test) for h in HEADS}
    anom = test.labels == 1
    for _ in range(N_RESAMPLES):
        counts = np.zeros(len(test.labels), dtype=int)
        for c in np.unique(test.cats):
            sel = np.flatnonzero(test.cats == c)
            counts[sel] = resample_counts(test.labels[sel], rng)
        for h in HEADS:
            reps[h].append(category_auroc(scores[h], test, counts))
            miss_reps[h].append(counts[anom] @ misses[h][anom] / counts[anom].sum())
    rows = []
    for h in HEADS:
        auc = category_auroc(scores[h], test)
        lo, hi = percentile_ci(np.array(reps[h]))
        dlo, dhi = percentile_ci(np.array(reps[h]) - np.array(reps["best_single"]))
        mlo, mhi = percentile_ci(np.array(miss_reps[h]))
        rows.append(
            {
                "test": split.test,
                "train": "+".join(split.train),
                "head": h,
                "chosen": str(chosen[h]),
                "image_auroc": auc,
                "ci_low": lo,
                "ci_high": hi,
                "diff_vs_best_single": auc - category_auroc(scores["best_single"], test),
                "diff_ci_low": dlo,
                "diff_ci_high": dhi,
                "miss_at_5pct_fpr": float(misses[h][anom].mean()),
                "miss_ci_low": mlo,
                "miss_ci_high": mhi,
            }
        )
    return rows, scores


def main() -> None:
    data = {d: Data(d) for d in ("mvtec", "visa", "bmad_brain", "mpdd", "headct")}
    rows = []
    for split in outer_splits():
        print(f"test {split.test:11s} train {'+'.join(split.train)}", flush=True)
        r, _ = evaluate(split, data)
        rows += r
        for x in r:
            print(
                f"   {x['head']:13s} {100 * x['image_auroc']:5.1f}"
                f" [{100 * x['ci_low']:5.1f}, {100 * x['ci_high']:5.1f}]"
                f"  vs best single {100 * x['diff_vs_best_single']:+5.1f}  ({x['chosen']})",
                flush=True,
            )
    OUT.mkdir(exist_ok=True)
    with open(OUT / "fusion_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print("wrote", OUT / "fusion_image.csv")


if __name__ == "__main__":
    main()
