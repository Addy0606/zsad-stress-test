"""Build the Phase 4 benchmark tables from cached outputs, with one command:

    python eval/make_tables.py                  # 1,000 bootstrap resamples (a few minutes)
    python eval/make_tables.py --n-resamples 100

Reads configs/bench_sources.yaml, and for each (method, dataset) the run fetched into
outputs/kaggle/<kernel>/outputs/<run>/ (scores.csv, metrics.json, run_info.json, config.yaml) and the
pixel statistics in outputs/kaggle/<stats_kernel>/stats/<method>/<dataset>/pixel_stats.npz.

Point estimates are the exact metrics (metrics.json, eval/metrics.py). The 95% CIs are percentile
bootstrap intervals (eval/bootstrap.py). Writes:
    results/benchmark.csv   one row per method x dataset x metric: value, CI, n_images, source run
    results/benchmark.md    one table per metric, plus the cost table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.bootstrap import METRICS, bootstrap_metrics, percentile_ci  # noqa: E402
from eval.cache import read_scores  # noqa: E402
from eval.pixel_stats import load_stats  # noqa: E402

METRIC_NAMES = {
    "image_auroc": "Image AUROC",
    "image_f1_max": "Image F1-max",
    "pixel_auroc": "Pixel AUROC",
    "aupro": "AUPRO (FPR ≤ 0.3)",
}
DATASET_NAMES = {
    "mvtec": "MVTec AD",
    "visa": "VisA",
    "mpdd": "MPDD",
    "bmad_brain": "BMAD Brain MRI",
    "headct": "Head CT",
}


def collect(manifest, kaggle_dir: Path, n_resamples: int, seed: int) -> list[dict]:
    rows = []
    stats_root = kaggle_dir / manifest.stats_kernel / "stats"
    for method, by_dataset in manifest.runs.items():
        for dataset, src in by_dataset.items():
            rdir = kaggle_dir / src.kernel / "outputs" / src.run
            source = f"{src.kernel}:{src.run}"
            if not (rdir / "scores.csv").exists():
                print(f"MISSING  {method:18s} {dataset:11s} {rdir}")
                continue
            exact = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
            stats_file = stats_root / method / dataset / "pixel_stats.npz"
            stats = load_stats(stats_file) if stats_file.exists() else None
            if stats is None and dataset != "headct":
                print(f"no pixel stats yet for {method} on {dataset}: pixel CIs left empty")
            reps = bootstrap_metrics(read_scores(rdir), stats, n_resamples=n_resamples, seed=seed)
            for m in METRICS:
                lo, hi = percentile_ci(reps[m])
                rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "metric": m,
                        "value": exact["mean"][m],
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_images": exact["n_images"],
                        "complete": exact["complete"],
                        "source": source,
                    }
                )
            print(f"done     {method:18s} {dataset:11s} ({exact['n_images']} images)", flush=True)
    return rows


def cost_rows(manifest, kaggle_dir: Path) -> list[dict]:
    """Per method: image-weighted mean seconds per image and max peak GPU memory, over all its runs."""
    out = []
    for method, by_dataset in manifest.runs.items():
        n = secs = 0.0
        peak, gpus = 0, set()
        for src in by_dataset.values():
            info_file = kaggle_dir / src.kernel / "outputs" / src.run / "run_info.json"
            if not info_file.exists():
                continue
            for s in json.loads(info_file.read_text(encoding="utf-8"))["sessions"]:
                n += s["images"]
                secs += s["images"] * s["mean_seconds_per_image"]
                peak = max(peak, s["peak_gpu_mb"] or 0)
                gpus.add(s["gpu"])
        if n:
            out.append(
                {
                    "method": method,
                    "images": int(n),
                    "s_per_image": secs / n,
                    "peak_gpu_mb": peak,
                    "gpu": "/".join(sorted(gpus)),
                }
            )
    return out


def fmt(r: dict | None) -> str:
    if r is None or math.isnan(r["value"]):
        return "—"
    ci = "" if math.isnan(r["ci_low"]) else f" [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"
    return f"{100 * r['value']:.1f}{ci}" + ("" if r["complete"] else " ⚠ incomplete")


def write_markdown(path: Path, manifest, rows: list[dict], costs: list[dict], n_resamples: int) -> None:
    index = {(r["method"], r["dataset"], r["metric"]): r for r in rows}
    datasets = list(manifest.datasets)
    single = [m for m in manifest.methods if not manifest.methods[m].get("transductive", False)]
    trans = [m for m in manifest.methods if manifest.methods[m].get("transductive", False)]
    lines = [
        "# Benchmark results (Phase 4)",
        "",
        "Generated by `python eval/make_tables.py` from `configs/bench_sources.yaml`; do not edit by hand.",
        f"Values in %: mean over categories, with a 95% percentile-bootstrap CI over images ({n_resamples} resamples,",
        "stratified by category and label). Head CT has image labels only. B2 is transductive (it uses the",
        "other unlabelled test images), so it is listed separately and never ranked with single-image methods.",
        "",
    ]
    for m in METRICS:
        lines += [f"## {METRIC_NAMES[m]}", "", "| Method | " + " | ".join(DATASET_NAMES[d] for d in datasets) + " |"]
        lines.append("|---|" + "---|" * len(datasets))
        for group in (single, trans):
            for meth in group:
                cells = [fmt(index.get((meth, d, m))) for d in datasets]
                lines.append(f"| {manifest.methods[meth].label} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Cost (inference only)",
        "",
        "| Method | Images timed | Seconds / image | Peak GPU memory (MB) | GPU |",
        "|---|---|---|---|---|",
    ]
    for c in costs:
        label = manifest.methods[c["method"]].label
        lines.append(f"| {label} | {c['images']} | {c['s_per_image']:.3f} | {c['peak_gpu_mb']} | {c['gpu']} |")
    lines += [
        "",
        "Seconds / image: time of one `predict` call (model forward + map post-processing, GPU synchronised), "
        "image-weighted over all datasets. Excludes data loading and, for B2, the per-category reference "
        "preparation. VisualAD's one-off training of the MVTec checkpoint is not included.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle-dir", default=str(REPO / "outputs" / "kaggle"))
    p.add_argument("--out", default=str(REPO / "results"))
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    rows = collect(manifest, Path(a.kaggle_dir), a.n_resamples, a.seed)
    costs = cost_rows(manifest, Path(a.kaggle_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    write_markdown(out / "benchmark.md", manifest, rows, costs, a.n_resamples)
    n_runs = len({(r["method"], r["dataset"]) for r in rows})
    print(f"wrote {out / 'benchmark.md'} and benchmark.csv ({n_runs} runs)")


if __name__ == "__main__":
    main()
