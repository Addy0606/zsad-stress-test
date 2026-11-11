"""Download the cached results (scripts/publish_cache.py) into outputs/kaggle/, the layout every script reads.

    python scripts/fetch_cache.py

No account needed; about 130 MB. After this, `python scripts/reproduce.py` rebuilds every table and figure.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
DATASET_ID = "addy0606/zsad-stress-test-cache"


def main() -> None:
    dest = REPO / "outputs" / "kaggle"
    snapshot_download(DATASET_ID, repo_type="dataset", local_dir=dest, ignore_patterns=["README.md", ".gitattributes"])
    print("cached results in", dest)


if __name__ == "__main__":
    main()
