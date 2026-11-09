"""Check that every dataset root in a paths profile exists and loads.
If a root is missing (e.g. Kaggle mounted a dataset under a different folder), search a base
folder for the dataset's marker file and print the root to put in configs/paths/<profile>.yaml.

Usage:
    python scripts/check_paths.py --paths kaggle --search /kaggle/input
    python scripts/check_paths.py --paths local
    python scripts/check_paths.py --paths kaggle --search /kaggle/input --fix   # rewrite missing roots

--fix rewrites configs/paths/<profile>.yaml with every root it finds, so later scripts that use
the profile see the real mount points (Kaggle has mounted datasets at more than one layout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.registry import build_dataset, dataset_names, load_dataset_config  # noqa: E402

# A file or folder that only exists at the dataset root (relative to the root).
MARKERS = {
    "mvtec": "bottle/ground_truth",
    "visa": "split_csv/1cls.csv",
    "mpdd": "bracket_black/ground_truth",
    "bmad_brain": "test/Ungood/label",
    "headct": "head_ct/head_ct",
}


def find_root(name: str, base: Path) -> Path | None:
    marker = MARKERS[name]
    last = marker.split("/")[-1]
    depth = marker.count("/")
    for hit in base.rglob(last):
        if hit.as_posix().endswith(marker):
            return hit.parents[depth]
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", default="kaggle")
    p.add_argument("--search", default=None, help="folder to search when a root is missing")
    p.add_argument("--fix", action="store_true", help="write found roots back into the paths profile")
    a = p.parse_args()

    ok = True
    fixes: dict[str, str] = {}
    for name in dataset_names():
        root = Path(load_dataset_config(name, a.paths).root)
        if root.exists():
            ds = build_dataset(name, a.paths)
            print(f"[ok]      {name:11s} {len(ds):5d} test images  ({root})")
            continue
        ok = False
        found = find_root(name, Path(a.search)) if a.search else None
        hint = f"found at {found} -> set `{name}: {found}` in configs/paths/{a.paths}.yaml" if found else "not found"
        print(f"[missing] {name:11s} {root}: {hint}")
        if found:
            fixes[name] = found.as_posix()
    if a.fix and fixes:
        write_fixes(a.paths, fixes)
    sys.exit(0 if ok else 1)


def write_fixes(profile: str, fixes: dict[str, str]) -> None:
    """Replace the `name: root` lines of configs/paths/<profile>.yaml for the datasets in `fixes`."""
    path = Path(__file__).resolve().parents[1] / "configs" / "paths" / f"{profile}.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        key = line.split(":", 1)[0].strip()
        if key in fixes:
            lines[i] = f"{key}: {fixes[key]}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {path} for: {', '.join(sorted(fixes))}")


if __name__ == "__main__":
    main()
