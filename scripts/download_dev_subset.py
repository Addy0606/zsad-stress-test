"""Download a small, fixed dev subset of each dataset from Kaggle (a few hundred MB at most),
so loaders, tests and sample grids can run on a laptop. Full datasets stay on Kaggle.

Needs a Kaggle API token in ~/.kaggle/kaggle.json.

Usage:
    python scripts/download_dev_subset.py                # all datasets
    python scripts/download_dev_subset.py --only visa    # one dataset
    python scripts/download_dev_subset.py --dest D:/Projects/data_dev

The subset is deterministic (fixed file names), so everyone gets the same files.
For VisA, a filtered copy of split_csv/1cls.csv listing only the downloaded images is written,
otherwise the loader would look for images that aren't there.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
import zipfile
from pathlib import Path

KAGGLE_REFS = {
    "mvtec": "ipythonx/mvtec-ad",
    "visa": "tensura3607/amazon-visa-anomaly",
    "mpdd": "alirezasalehy/mpdd-ad",
    "headct": "felipekitamura/head-ct-hemorrhage",
    "bmad": "duongkaiba/bmad-aaclip",
}
N = 4  # images per (category, normal/defect type) in the subset


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def fetch(api, ref: str, file_name: str, dest_root: Path) -> Path:
    """Download one file to dest_root/file_name (skips if present). Retries on rate limits."""
    target = dest_root / file_name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            # The API recreates file_name's folders under `path`, so pass the dataset root.
            api.dataset_download_file(ref, file_name, path=str(dest_root), quiet=True)
            break
        except Exception as e:  # mostly HTTP 429 (rate limit)
            if attempt == 5:
                raise
            wait = 10 * (attempt + 1)
            print(f"  retry in {wait}s ({e.__class__.__name__}: {str(e)[:80]})")
            time.sleep(wait)
    # Kaggle sometimes delivers a single file zipped as <name>.zip
    zipped = target.parent / (target.name + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            z.extractall(target.parent)
        zipped.unlink()
    if not target.exists():
        raise FileNotFoundError(f"download of {file_name} did not produce {target}")
    return target


def mvtec_layout_files(spec: dict[str, list[str]]) -> list[str]:
    """spec: category -> defect types. Takes N normal images, N per defect type, and their masks."""
    files = []
    for cat, defects in spec.items():
        files += [f"{cat}/test/good/{i:03d}.png" for i in range(N)]
        for d in defects:
            for i in range(N):
                files += [f"{cat}/test/{d}/{i:03d}.png", f"{cat}/ground_truth/{d}/{i:03d}_mask.png"]
    return files


def download_mvtec(api, dest: Path) -> None:
    spec = {"bottle": ["broken_large"], "screw": ["scratch_head"], "carpet": ["hole"]}
    for f in mvtec_layout_files(spec):
        fetch(api, KAGGLE_REFS["mvtec"], f, dest / "mvtec")


def download_mpdd(api, dest: Path) -> None:
    spec = {"bracket_black": ["hole"], "metal_plate": ["scratches"]}
    for f in mvtec_layout_files(spec):
        fetch(api, KAGGLE_REFS["mpdd"], f, dest / "mpdd")


def download_visa(api, dest: Path) -> None:
    root, ref = dest / "visa", KAGGLE_REFS["visa"]
    split = fetch(api, ref, "split_csv/1cls.csv", root / "_full")
    with open(split, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    keep = []
    for cat in ["candle", "pcb1"]:
        for label in ["normal", "anomaly"]:
            keep += [r for r in rows if r["object"] == cat and r["label"] == label][:N]
        fetch(api, ref, f"{cat}/image_anno.csv", root)
    for r in keep:
        fetch(api, ref, r["image"], root)
        if r["mask"]:
            fetch(api, ref, r["mask"], root)
    (root / "split_csv").mkdir(parents=True, exist_ok=True)
    with open(root / "split_csv" / "1cls.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["object", "split", "label", "image", "mask"])
        w.writeheader()
        w.writerows(keep)
    shutil.rmtree(root / "_full")


def download_headct(api, dest: Path) -> None:
    """Head CT is only 25 MB, so take all of it."""
    root = dest / "headct"
    if (root / "labels.csv").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_REFS["headct"], path=str(root), unzip=True, quiet=True)


def download_bmad(api, dest: Path) -> None:
    """Only the Brain test split is used (640 normal + 3,075 tumour slices on Kaggle).
    File names picked from the dataset listing, spread across patients."""
    base = "Brain_AD/BraTS2021_slice/test"
    normal = ["00000_96.png", "00457_98.png", "00805_62.png", "01244_64.png"]
    tumour = ["00002_60.png", "00481_60.png", "01090_60.png", "01406_60.png"]
    files = [f"good/img/{n}" for n in normal]
    files += [f"Ungood/{sub}/{n}" for n in tumour for sub in ("img", "label")]
    for f in files:
        fetch(api, KAGGLE_REFS["bmad"], f"{base}/{f}", dest / "bmad")


DOWNLOADERS = {
    "mvtec": download_mvtec,
    "visa": download_visa,
    "mpdd": download_mpdd,
    "headct": download_headct,
    "bmad": download_bmad,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="D:/Projects/data_dev")
    p.add_argument("--only", nargs="*", choices=sorted(DOWNLOADERS), default=None)
    a = p.parse_args()
    dest = Path(a.dest)
    api = get_api()
    for name in a.only or sorted(DOWNLOADERS):
        print(f"== {name}")
        DOWNLOADERS[name](api, dest)
    print("done")


if __name__ == "__main__":
    main()
