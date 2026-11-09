"""Clone the third-party method repos at pinned commits into third_party/ (gitignored),
and download the VisualAD checkpoint (stored with Git LFS, so a plain clone only gets a pointer).

Their code is never copied into this repo: AnomalyVFM and VisualAD have no licence file,
so we only call them. AnomalyCLIP is MIT.

Usage:
    python scripts/setup_third_party.py
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party"

REPOS = {
    # name: (url, commit checked in Phase 0)
    "AnomalyCLIP": ("https://github.com/zqhang/AnomalyCLIP.git", "3911738c0867544f545a076ad78f3f11d9ecbfdf"),
    "VisualAD": ("https://github.com/7HHHHH/VisualAD.git", "ba7a866e62eebeeda52a9f85f4d8388f691042d8"),
    "AnomalyVFM": ("https://github.com/MaticFuc/AnomalyVFM.git", "4da96b493a0b12e4e380fa5bf6573c126342855c"),
}

VISUALAD_WEIGHT = (
    "https://media.githubusercontent.com/media/7HHHHH/VisualAD/"
    "ba7a866e62eebeeda52a9f85f4d8388f691042d8/weight/train_on_visa/CLIP.pth",
    "VisualAD/weight/train_on_visa/CLIP.pth",
    126_119_090,  # expected size in bytes (from the LFS pointer)
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    for name, (url, commit) in REPOS.items():
        dest = ROOT / name
        if not dest.exists():
            run(["git", "clone", url, str(dest)])
        run(["git", "fetch", "--quiet", "origin", commit], cwd=dest)
        run(["git", "checkout", "--quiet", commit], cwd=dest)

    url, rel, size = VISUALAD_WEIGHT
    target = ROOT / rel
    if not target.exists() or target.stat().st_size != size:
        print(f"downloading VisualAD checkpoint -> {target}")
        urllib.request.urlretrieve(url, target)
    if target.stat().st_size != size:
        raise RuntimeError(f"{target} has {target.stat().st_size} bytes, expected {size}")
    print("third-party setup done")


if __name__ == "__main__":
    main()
