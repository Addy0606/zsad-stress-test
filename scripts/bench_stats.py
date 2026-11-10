"""Compute per-image pixel statistics (eval/pixel_stats.py) for every run in configs/bench_sources.yaml
that has pixel labels. Runs on Kaggle as a CPU job with the benchmark kernels' outputs attached:

    python scripts/kaggle_submit.py --name zsad-bench-stats --cpu --datasets mvtec visa mpdd bmad_brain \
        --kernel-sources <kernels from the manifest> --pre "python scripts/bench_stats.py"

Writes /kaggle/working/stats/<method>/<dataset>/pixel_stats.npz (+ _check.json). Already-written runs
are skipped, and one failing run doesn't stop the others.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.registry import load_dataset_config  # noqa: E402
from eval.pixel_stats import run_pixel_stats  # noqa: E402

PIXEL_DATASETS = {"mvtec", "visa", "mpdd", "bmad_brain"}  # headct has image labels only


def find_run(base: Path, kernel: str, run: str) -> Path | None:
    suffix = f"{kernel}/outputs/{run}"
    for hit in base.rglob("scores.csv"):
        if hit.parent.as_posix().endswith(suffix):
            return hit.parent
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/stats")
    p.add_argument("--paths", default="kaggle")
    p.add_argument("--only", nargs="*", default=None, help="method:dataset pairs (default: all)")
    a = p.parse_args()

    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    failed = []
    for method, by_dataset in manifest.runs.items():
        for dataset, src in by_dataset.items():
            if dataset not in PIXEL_DATASETS or (a.only and f"{method}:{dataset}" not in a.only):
                continue
            out = Path(a.out) / method / dataset
            if (out / "pixel_stats.npz").exists():
                continue
            rdir = find_run(Path(a.input), src.kernel, src.run)
            print(f"===== {method} on {dataset}: {rdir}", flush=True)
            if rdir is None:
                failed.append((method, dataset, "run not found"))
                continue
            try:
                load_dataset_config(dataset, a.paths)
                run_pixel_stats(rdir, dataset, a.paths, out)
            except Exception as e:  # keep going; report at the end
                traceback.print_exc()
                failed.append((method, dataset, repr(e)))
    print("FAILED:", failed if failed else "none")


if __name__ == "__main__":
    main()
