"""Publish the cached results every table and figure is built from, as a public Hugging Face dataset (Phase 8).

What goes up: per-image scores, metric files, run configs and per-image pixel histograms from the Kaggle jobs
listed in KERNELS, in the same folder layout as outputs/kaggle/. What doesn't: images, anomaly maps, model
weights (our VisualAD checkpoint is derived from an unlicensed repo) and job logs.
Licence: CC BY-NC-SA 4.0, the strictest licence among the source datasets (MVTec AD, MPDD, BMAD).

    python scripts/publish_cache.py              # list what would be uploaded
    python scripts/publish_cache.py --upload     # needs `huggingface-cli login`
Download side: scripts/fetch_cache.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "kaggle"
DATASET_ID = "addy0606/zsad-stress-test-cache"
KERNELS = (  # every Kaggle job whose outputs a table or figure reads
    "zsad-repro-anomalyclip-l24",
    "zsad-repro-visualad",
    "zsad-repro-anomalyvfm",
    "zsad-repro-baselines",
    "zsad-bench-clipfam",
    "zsad-bench-vfmfam",
    "zsad-bench-stats",
    "zsad-p5-regions",
    "zsad-p5-mapscores",
    "zsad-p6-pixelfusion",
)
SUFFIXES = (".csv", ".json", ".yaml", ".npz")

CARD = """---
license: cc-by-nc-sa-4.0
pretty_name: Zero-shot anomaly detection stress test, cached results
tags: [anomaly-detection, zero-shot, benchmark]
---

Cached outputs of [zsad-stress-test](https://github.com/Addy0606/zsad-stress-test): per-image scores, metrics,
run configs and per-image pixel-level histograms for AnomalyCLIP, VisualAD, AnomalyVFM (DINOv2, RADIO) and two
baselines on MVTec AD, VisA, MPDD, BMAD Brain MRI and Head CT, plus the failure-analysis and fusion outputs.
No images, anomaly maps or model weights.

Download with `python scripts/fetch_cache.py` from the code repo, then rebuild every table and figure with
`python scripts/reproduce.py`.

Derived from MVTec AD (CC BY-NC-SA 4.0), VisA (CC BY 4.0), MPDD (CC BY-NC-SA 4.0), BMAD / BraTS 2021
(CC BY-NC-SA) and Head CT (CC0); hence CC BY-NC-SA 4.0. Please cite the original datasets.
"""


def cache_files() -> list[Path]:
    files = []
    for k in KERNELS:
        found = [p for p in (CACHE / k).rglob("*") if p.is_file() and p.suffix in SUFFIXES and "maps" not in p.parts]
        if not found:
            raise SystemExit(f"nothing cached for {k}: fetch it first (scripts/kaggle_fetch.py --name {k})")
        files += found
    return sorted(files)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    files = cache_files()
    size = sum(f.stat().st_size for f in files) / 2**20
    print(f"{len(files)} files, {size:.0f} MB from {len(KERNELS)} jobs")
    if not a.upload:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(DATASET_ID, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=DATASET_ID, repo_type="dataset")
    api.upload_folder(
        folder_path=str(CACHE),
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=[f.relative_to(CACHE).as_posix() for f in files],
        commit_message="Upload cached results",
    )
    print(f"uploaded: https://huggingface.co/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
