"""Submit benchmark jobs to Kaggle as a private GPU notebook (needs ~/.kaggle/kaggle.json and a
phone-verified account for GPU + internet).

Usage:
    python scripts/kaggle_submit.py --name zsad-anomalyclip --jobs anomalyclip:mvtec anomalyclip:visa
    python scripts/kaggle_submit.py --name zsad-repro --jobs visualad:mvtec --branch phase-2 --wait

The notebook (notebooks/kaggle_benchmark.ipynb) gets its parameters cell filled in, only the
datasets the jobs need are attached, and the kernel is pushed. Re-pushing the same --name creates
a new version. Fetch results afterwards with scripts/kaggle_fetch.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE_DATASETS = {
    "mvtec": "ipythonx/mvtec-ad",
    "visa": "tensura3607/amazon-visa-anomaly",
    "mpdd": "alirezasalehy/mpdd-ad",
    "bmad_brain": "duongkaiba/bmad-aaclip",
    "headct": "felipekitamura/head-ct-hemorrhage",
}


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def wait_for(api, ref: str, poll_s: int = 60) -> str:
    """Block until the kernel finishes; return its final status."""
    time.sleep(20)
    while True:
        status = str(api.kernels_status(ref).status)
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if any(s in status.upper() for s in ("COMPLETE", "ERROR", "CANCEL")):
            return status
        time.sleep(poll_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="kernel slug, e.g. zsad-anomalyclip")
    p.add_argument("--jobs", nargs="+", required=True, help="method:dataset pairs")
    p.add_argument("--branch", default="main")
    p.add_argument("--extra-args", default="", help="appended to every run_benchmark.py call")
    p.add_argument("--pre", nargs="*", default=[], help="shell commands to run before the jobs")
    p.add_argument("--wait", action="store_true", help="poll until the run finishes")
    a = p.parse_args()

    jobs = [tuple(j.split(":")) for j in a.jobs]
    for _method, d in jobs:
        if d not in KAGGLE_DATASETS:
            sys.exit(f"unknown dataset {d}")
    api = get_api()
    user = api.get_config_value("username")

    nb = json.loads((REPO / "notebooks" / "kaggle_benchmark.ipynb").read_text())
    params = next(c for c in nb["cells"] if "parameters" in c["metadata"].get("tags", []))
    params["source"] = (
        f"BRANCH = {a.branch!r}\nJOBS = {jobs!r}\nEXTRA_ARGS = {a.extra_args!r}\nPRE_COMMANDS = {a.pre!r}"
    )

    kdir = REPO / "outputs" / "kaggle_kernels" / a.name
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": f"{user}/{a.name}",
        "title": a.name,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": sorted({KAGGLE_DATASETS[d] for _, d in jobs}),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    api.kernels_push(str(kdir))
    print(f"pushed {user}/{a.name}: https://www.kaggle.com/code/{user}/{a.name}")
    if a.wait:
        print("final status:", wait_for(api, f"{user}/{a.name}"))


if __name__ == "__main__":
    main()
