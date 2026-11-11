"""Assemble the Hugging Face Space for the demo (Phase 7) and optionally upload it.

The Space gets only what the app needs: app.py at the root, the method wrappers and their configs, the fusion
normaliser, the third-party setup script and the two small data files in app/. No third-party code, weights or
datasets: the app downloads those from the original authors when it starts.

    python scripts/deploy_space.py                      # build outputs/space/ only, to inspect or run locally
    python scripts/deploy_space.py --upload --user addy0606   # needs `huggingface-cli login` first
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "outputs" / "space"
SPACE_NAME = "zsad-stress-test"
FILES = {  # Space path: repo path
    "app.py": "app/app.py",
    "requirements.txt": "app/requirements.txt",
    "app/fusion_normalizer.npz": "app/fusion_normalizer.npz",
    "app/score_reference.json": "app/score_reference.json",
    "fusion/__init__.py": "fusion/__init__.py",
    "fusion/normalize.py": "fusion/normalize.py",
    "scripts/setup_third_party.py": "scripts/setup_third_party.py",
}
GLOBS = ("methods/*.py", "configs/methods/*.yaml")

CARD = """---
title: Zero-Shot Anomaly Detection Stress Test
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
startup_duration_timeout: 1h
pinned: false
---

Demo for [zsad-stress-test](https://github.com/Addy0606/zsad-stress-test): four zero-shot anomaly detectors
(AnomalyCLIP, VisualAD, AnomalyVFM with DINOv2 and RADIO backbones) and their fused anomaly map on your own photo.
Runs on a free CPU, so one photo takes about a minute. Third-party code and weights are downloaded from the
original authors at startup; see the credits in the app.
"""


def build() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    pairs = dict(FILES)
    for pattern in GLOBS:
        for path in REPO.glob(pattern):
            rel = path.relative_to(REPO).as_posix()
            pairs[rel] = rel
    for dst, src in sorted(pairs.items()):
        (STAGE / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / src, STAGE / dst)
    (STAGE / "README.md").write_text(CARD, encoding="utf-8")
    print(f"built {STAGE} ({len(pairs) + 1} files)")


def upload(user: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = f"{user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo_id, repo_type="space", commit_message="Update demo")
    print(f"uploaded: https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true")
    p.add_argument("--user", default="addy0606")
    a = p.parse_args()
    build()
    if a.upload:
        upload(a.user)


if __name__ == "__main__":
    main()
