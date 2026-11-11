"""Pixel-level fusion of the four detectors' anomaly maps, leave-one-dataset-out (Phase 6).

Runs as a Kaggle CPU job with the benchmark kernels' outputs and the datasets attached (the maps live there):
    python fusion/pixel_fusion.py --out /kaggle/working/pixel_fusion

Per pixel, the inputs are the four detectors' map values at 256x256, each mapped through its training-data
distribution (fusion/normalize.py). Heads, fitted on the training datasets of a split only (fusion/splits.py):
- best_single  the detector with the highest mean AUPRO on the TRAINING datasets (from results/benchmark.csv)
- mean         average of the four normalised maps (no learning)
- logreg       logistic regression on the four normalised values; C chosen by inner leave-one-dataset-out,
               scored by pixel AUROC on the sampled validation pixels
Training pixels: from every image, up to 50 normal pixels, and from every anomalous image up to 50 defect pixels;
each dataset gets equal total weight and normal/defect are balanced within it.

For each held-out dataset and head we write, from the fused map of EVERY test image:
- metrics.json        exact pixel AUROC and AUPRO (mean over categories), as in eval/metrics.py
- pixel_stats.npz     per-image histograms for the bootstrap (eval/pixel_stats.py)
- image_scores.csv    an image score read off the fused map (mean of its top 1%), for the image-level comparison
Head CT has no masks: it only gets image_scores.csv (heads still trained on the three fold datasets).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from eval.cache import EVAL_SIZE, load_map, read_scores, relative_path, run_dataset_root  # noqa: E402
from eval.metrics import aupro, pixel_auroc  # noqa: E402
from eval.pixel_stats import compute_stats, save_stats  # noqa: E402
from fusion.normalize import QuantileNormalizer  # noqa: E402
from fusion.splits import FOLD_DATASETS, outer_splits  # noqa: E402
from methods.base import topk_mean  # noqa: E402

DETECTORS = ("anomalyclip", "visualad", "anomalyvfm_dinov2", "anomalyvfm_radio")
PIXEL_DATASETS = ("mvtec", "visa", "bmad_brain", "mpdd")
HEADS = ("best_single", "mean", "logreg")
LOGREG_C = (0.01, 0.1, 1.0, 10.0)
PER_IMAGE = 50
NORM_PER_IMAGE = 30
SEED = 0


class DatasetMaps:
    """The four detectors' maps for every image of one dataset, aligned by path relative to the dataset root."""

    def __init__(self, dataset_name: str, paths: str, input_root: Path, manifest) -> None:
        from bench_stats import find_run

        from data.registry import build_dataset

        self.name = dataset_name
        self.dataset = build_dataset(dataset_name, paths)
        self.sources = []  # per detector: {key: (rdir, stored path)}
        meta = None
        for m in DETECTORS:
            src = manifest.runs[m][dataset_name]
            rdir = find_run(input_root, src.kernel, src.run)
            root = run_dataset_root(rdir)
            rows = read_scores(rdir)
            self.sources.append({relative_path(r["path"], root): (rdir, r["path"]) for r in rows})
            if meta is None:
                meta = {relative_path(r["path"], root): (r["category"], int(r["label"])) for r in rows}
        self.keys = sorted(meta)
        self.cats = np.array([meta[k][0] for k in self.keys])
        self.labels = np.array([meta[k][1] for k in self.keys])
        index = {relative_path(r.image_path, self.dataset.root): i for i, r in enumerate(self.dataset.records)}
        self.index = [index[k] for k in self.keys]
        self.has_masks = self.dataset.has_pixel_labels

    def maps(self, i: int) -> np.ndarray:
        """(4, 256, 256) float32 maps of image i."""
        k = self.keys[i]
        return np.stack([load_map(*src[k]) for src in self.sources]).astype(np.float32)

    def mask(self, i: int) -> np.ndarray:
        return self.dataset.load_mask(self.index[i], EVAL_SIZE).astype(bool)


def sample_pixels(d: DatasetMaps, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Training pixels (X, y) stratified per image, plus uniform pixels for the normaliser."""
    xs, ys, us = [], [], []
    for i in range(len(d.keys)):
        m = d.maps(i).reshape(4, -1).T  # (65536, 4)
        mask = d.mask(i).ravel() if d.has_masks else np.zeros(len(m), dtype=bool)
        neg, pos = np.flatnonzero(~mask), np.flatnonzero(mask)
        pick_n = rng.choice(neg, min(PER_IMAGE, len(neg)), replace=False)
        pick_p = rng.choice(pos, min(PER_IMAGE, len(pos)), replace=False) if len(pos) else np.array([], dtype=int)
        xs += [m[pick_n], m[pick_p]]
        ys += [np.zeros(len(pick_n)), np.ones(len(pick_p))]
        us.append(m[rng.choice(len(m), NORM_PER_IMAGE, replace=False)])
    return {"X": np.vstack(xs), "y": np.concatenate(ys), "U": np.vstack(us)}


def weights(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))


def fit_normalizer(samples: list[dict], rng: np.random.Generator) -> QuantileNormalizer:
    n = min(len(s["U"]) for s in samples)
    return QuantileNormalizer().fit(np.vstack([s["U"][rng.choice(len(s["U"]), n, replace=False)] for s in samples]))


def fit_logreg(samples: list[dict], norm: QuantileNormalizer, c: float) -> LogisticRegression:
    X = np.vstack([norm.transform(s["X"]) for s in samples])
    y = np.concatenate([s["y"] for s in samples])
    w = np.concatenate([weights(s["y"]) for s in samples])
    return LogisticRegression(C=c, max_iter=2000).fit(X, y, sample_weight=w)


def choose_c(train: tuple[str, ...], samples: dict[str, dict], inner) -> float:
    rng = np.random.default_rng(SEED)
    best, best_auc = None, -1.0
    for c in LOGREG_C:
        aucs = []
        for tr, v in inner:
            norm = fit_normalizer([samples[t] for t in tr], rng)
            model = fit_logreg([samples[t] for t in tr], norm, c)
            sv = samples[v]
            aucs.append(
                roc_auc_score(sv["y"], model.decision_function(norm.transform(sv["X"])), sample_weight=weights(sv["y"]))
            )
        if np.mean(aucs) > best_auc:
            best, best_auc = c, float(np.mean(aucs))
    return best


def best_single_by_training_aupro(train: tuple[str, ...]) -> int:
    with open(REPO / "results" / "benchmark.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["metric"] == "aupro"]
    means = [
        np.mean([float(r["value"]) for r in rows if r["method"] == m and r["dataset"] in train]) for m in DETECTORS
    ]
    return int(np.argmax(means))


def run_split(split, data: dict[str, DatasetMaps], samples: dict[str, dict], out: Path) -> dict:
    rng = np.random.default_rng(SEED)
    norm = fit_normalizer([samples[t] for t in split.train], rng)
    best = best_single_by_training_aupro(split.train)
    c = choose_c(split.train, samples, split.inner())
    model = fit_logreg([samples[t] for t in split.train], norm, c)
    heads = {
        "best_single": lambda m: m[best],
        "mean": lambda m: norm.transform(m.reshape(4, -1).T).mean(1).reshape(m.shape[1:]),
        "logreg": lambda m: model.decision_function(norm.transform(m.reshape(4, -1).T)).reshape(m.shape[1:]),
    }
    d = data[split.test]
    info = {
        "test": split.test,
        "train": list(split.train),
        "best_single": DETECTORS[best],
        "logreg_C": c,
        "logreg_coef": dict(zip(DETECTORS, model.coef_[0].round(4).tolist(), strict=True)),
    }
    print(json.dumps(info), flush=True)
    for h, fn in heads.items():
        hdir = out / split.test / h
        hdir.mkdir(parents=True, exist_ok=True)
        rows, per_cat_stats, per_cat_paths, exact = [], {}, {}, {}
        for cat in sorted(set(d.cats)):
            idx = np.flatnonzero(d.cats == cat)
            fused = np.stack([fn(d.maps(i)) for i in idx]).astype(np.float32)
            for j, i in enumerate(idx):
                rows.append(
                    {"image": d.keys[i], "category": cat, "label": int(d.labels[i]), "score": topk_mean(fused[j], 0.01)}
                )
            if d.has_masks:
                masks = np.stack([d.mask(i) for i in idx])
                if masks.any():
                    exact[cat] = {"pixel_auroc": pixel_auroc(masks, fused), "aupro": aupro(masks, fused)}
                    per_cat_stats[cat] = compute_stats(masks, fused)
                    per_cat_paths[cat] = [d.keys[i] for i in idx]
        with open(hdir / "image_scores.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if exact:
            save_stats(hdir / "pixel_stats.npz", per_cat_stats, per_cat_paths)
            mean = {k: float(np.mean([v[k] for v in exact.values()])) for k in ("pixel_auroc", "aupro")}
            (hdir / "metrics.json").write_text(json.dumps({"mean": mean, "per_category": exact, **info}, indent=2))
            print(
                f"   {h:12s} pixel AUROC {100 * mean['pixel_auroc']:.1f}  AUPRO {100 * mean['aupro']:.1f}", flush=True
            )
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/pixel_fusion")
    p.add_argument("--paths", default="kaggle")
    a = p.parse_args()

    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    out = Path(a.out)
    rng = np.random.default_rng(SEED)
    data, samples = {}, {}
    for name in (*PIXEL_DATASETS, "headct"):
        print(f"===== loading {name}", flush=True)
        data[name] = DatasetMaps(name, a.paths, Path(a.input), manifest)
        if name in FOLD_DATASETS:  # only these ever provide training pixels (MPDD and Head CT are test-only)
            samples[name] = sample_pixels(data[name], rng)
            print(f"   {len(samples[name]['y'])} training pixels ({int(samples[name]['y'].sum())} defect)", flush=True)
    choices, failed = [], []
    for split in outer_splits():
        print(f"===== test {split.test} (train {'+'.join(split.train)})", flush=True)
        try:
            choices.append(run_split(split, data, samples, out))
        except Exception as e:
            traceback.print_exc()
            failed.append((split.test, repr(e)))
    (out / "choices.json").write_text(json.dumps(choices, indent=2))
    print("FAILED:", failed if failed else "none")


if __name__ == "__main__":
    main()
