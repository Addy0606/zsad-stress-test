"""Rebuild every table and figure from the cached results, in one command (Phase 8).

    python scripts/fetch_cache.py     # once: download the cached results (~130 MB) into outputs/kaggle/
    python scripts/reproduce.py       # then rebuild everything; the git diff afterwards should be empty

Each step is an ordinary script that can also be run on its own. Nothing here touches a GPU or the datasets:
the GPU work (running the detectors, Kaggle jobs) produced the cache, and this turns the cache into results.
Bootstrap CIs use fixed seeds, so the rebuilt files match the committed ones exactly.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEPS = (  # (script, what it writes)
    ("eval/make_tables.py", "results/benchmark.{csv,md}: main table with 95% CIs, cost table"),
    ("analysis/image_level.py", "analysis/results/*.csv: complementarity, defect types, domain drops"),
    ("analysis/region_level.py", "analysis/results/*.csv: peak-in-mask, regions found by size / contrast"),
    ("analysis/figures.py", "analysis/figures/*.png: failure-analysis figures"),
    ("fusion/image_fusion.py", "results/fusion_image.csv: image-level fusion heads, leave-one-dataset-out"),
    ("fusion/report.py", "results/fusion*.{csv,md}: pixel-level fusion and the ablation tables"),
    ("app/build_reference.py", "app/score_reference.json: the demo's normal-image reference scores"),
    ("report/figures.py", "report/figures/*.png: overview and fusion figures for the README and report"),
    ("report/readme_tables.py", "README.md: result tables copied from the generated files"),
)


def main() -> None:
    missing = [k for k in ("zsad-bench-stats", "zsad-p6-pixelfusion") if not (REPO / "outputs" / "kaggle" / k).exists()]
    if missing:
        raise SystemExit(f"cached results missing ({', '.join(missing)}): run `python scripts/fetch_cache.py` first")
    t_all = time.perf_counter()
    for script, writes in STEPS:
        print(f"\n===== {script}  ->  {writes}", flush=True)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(REPO / script)], cwd=REPO, check=True)
        print(f"===== {script} done in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nall {len(STEPS)} steps done in {(time.perf_counter() - t_all) / 60:.1f} min")
    print("check: `git status` should show no changes to results/, analysis/ or app/")


if __name__ == "__main__":
    main()
