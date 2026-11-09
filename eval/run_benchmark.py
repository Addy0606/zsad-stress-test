"""Run one method on one dataset and cache the outputs per image (resumable).

Usage:
    python eval/run_benchmark.py --method anomalyclip --dataset mvtec --paths kaggle
    python eval/run_benchmark.py --method visualad --dataset visa --paths local --limit 8
    python eval/run_benchmark.py --method anomalyclip --dataset mvtec --method-override params.features_list=[24]

If the session drops, run the same command again: finished images are skipped.
Afterwards `eval/evaluate.py` computes metrics from the cache (also run automatically at the end).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data.registry import build_dataset, load_dataset_config  # noqa: E402
from eval.cache import EVAL_SIZE, append_score, done_paths, run_dir, save_map  # noqa: E402
from methods.base import resize_map  # noqa: E402
from methods.registry import build_method, resolve_method_config  # noqa: E402


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


KEY_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pillow",
    "timm",
    "open_clip_torch",
    "transformers",
    "huggingface_hub",
    "omegaconf",
    "opencv-python-headless",
    "opencv-python",
)


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


class LazyImages(Sequence):
    """Images of a dataset subset, loaded only when indexed (for transductive methods)."""

    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset, self.indices = dataset, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.dataset[self.indices[i]]["image"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--paths", default="local")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--run-name", default=None, help="output folder name (default: method name)")
    p.add_argument("--method-override", nargs="*", default=[], help="e.g. params.topk_ratio=0.05")
    p.add_argument("--dataset-override", nargs="*", default=[], help="e.g. categories=[bottle]")
    p.add_argument("--limit", type=int, default=None, help="only the first N images (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-eval", action="store_true")
    a = p.parse_args()

    set_seeds(a.seed)
    ds_cfg = load_dataset_config(a.dataset, a.paths, a.dataset_override)
    dataset = build_dataset(a.dataset, a.paths, a.dataset_override)
    m_cfg = resolve_method_config(a.method, target=a.dataset, domain=dataset.domain, overrides=a.method_override)
    rdir = run_dir(a.out, a.run_name or m_cfg.name, a.dataset)
    rdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OmegaConf.save(
        OmegaConf.create(
            {
                "method": OmegaConf.to_container(m_cfg),
                "dataset": OmegaConf.to_container(ds_cfg),
                "seed": a.seed,
                "limit": a.limit,
                "eval_size": EVAL_SIZE,
                "git_commit": git_commit(),
                "torch": str(torch.__version__),
                "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "packages": package_versions(),
            }
        ),
        rdir / "config.yaml",
    )

    indices = list(range(len(dataset)))[: a.limit]
    finished = done_paths(rdir)
    todo = [i for i in indices if dataset.records[i].image_path not in finished]
    print(f"{m_cfg.name} on {a.dataset}: {len(indices)} images, {len(indices) - len(todo)} cached, {len(todo)} to run")

    times: list[float] = []
    if todo:
        method = build_method(m_cfg)
        method.setup(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        by_category: dict[str, list[int]] = {}
        for i in todo:
            by_category.setdefault(dataset.records[i].category, []).append(i)
        for category, cat_todo in by_category.items():
            if method.transductive:
                # references come from ALL test images of the category, not just the unfinished ones
                cat_all = [i for i in indices if dataset.records[i].category == category]
                method.prepare_category(LazyImages(dataset, cat_all), category)
            for n, i in enumerate(cat_todo):
                item = dataset[i]
                t0 = time.perf_counter()
                score, amap = method.predict(item["image"], item["category"])
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                times.append(dt)
                save_map(rdir, item["path"], resize_map(amap, (EVAL_SIZE, EVAL_SIZE)))
                append_score(rdir, {**item, "score": score, "seconds": round(dt, 4)})
                if n % 50 == 0:
                    print(f"  {category}: {n + 1}/{len(cat_todo)}  {dt:.2f}s/img", flush=True)

        info_file = rdir / "run_info.json"
        info = json.loads(info_file.read_text()) if info_file.exists() else {"sessions": []}
        info["sessions"].append(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "images": len(times),
                "mean_seconds_per_image": float(np.mean(times)),
                "median_seconds_per_image": float(np.median(times)),
                "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            }
        )
        info_file.write_text(json.dumps(info, indent=2))

    if not a.no_eval:
        from eval.evaluate import evaluate_run

        evaluate_run(rdir, a.dataset, a.paths, a.dataset_override, a.limit)


if __name__ == "__main__":
    main()
