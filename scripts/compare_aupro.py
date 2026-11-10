"""Score the same cached anomaly maps with our AUPRO and with AnomalyCLIP's official `cal_pro_score`,
to check whether a gap to a published AUPRO comes from the metric implementation.

Ours (eval/metrics.py): exact PRO curve over all thresholds, integrated to FPR = 0.3, divided by 0.3.
Theirs (third_party/AnomalyCLIP/metrics.py): 200 evenly spaced thresholds, keeps points with FPR < 0.3,
then min-max rescales those FPRs to [0, 1] before integrating.

Usage (needs the full dataset, so normally on Kaggle):
    python scripts/compare_aupro.py --run <outputs>/anomalyclip_layer24/mvtec --dataset mvtec --paths kaggle
    python scripts/compare_aupro.py --find anomalyclip_layer24/mvtec --dataset mvtec --paths kaggle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.registry import build_dataset  # noqa: E402
from eval.cache import EVAL_SIZE, load_map, match_records, read_scores  # noqa: E402
from eval.metrics import aupro  # noqa: E402


def official_pro():
    sys.path.insert(0, str(REPO / "third_party" / "AnomalyCLIP"))
    from metrics import cal_pro_score  # noqa: E402

    return cal_pro_score


def find_run(suffix: str, base: Path = Path("/kaggle/input")) -> Path:
    for hit in base.rglob("scores.csv"):
        if hit.parent.as_posix().endswith(suffix):
            return hit.parent
    sys.exit(f"no run ending in {suffix} under {base}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--find", default=None, help="search /kaggle/input for a run dir ending in this")
    p.add_argument("--dataset", required=True)
    p.add_argument("--paths", default="kaggle")
    p.add_argument("--out", default="/kaggle/working/aupro_compare")
    a = p.parse_args()

    rdir = Path(a.run) if a.run else find_run(a.find)
    print("run:", rdir, flush=True)
    cal_pro_score = official_pro()
    dataset = build_dataset(a.dataset, a.paths)
    rows = read_scores(rdir)
    index_of = match_records(dataset, rdir, rows)

    result = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        masks = np.stack([dataset.load_mask(index_of[r["path"]], EVAL_SIZE) for r in cat_rows]).astype(np.uint8)
        maps = np.stack([load_map(rdir, r["path"]) for r in cat_rows])
        ours, theirs = aupro(masks, maps), float(cal_pro_score(masks, maps))
        result[cat] = {"ours": ours, "official": theirs}
        print(f"{cat:16s} ours {100 * ours:5.1f}   official {100 * theirs:5.1f}", flush=True)
    mean = {k: float(np.mean([v[k] for v in result.values()])) for k in ("ours", "official")}
    print(f"{'MEAN':16s} ours {100 * mean['ours']:5.1f}   official {100 * mean['official']:5.1f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.dataset}.json").write_text(
        json.dumps({"run": str(rdir), "mean": mean, "per_category": result}, indent=2)
    )


if __name__ == "__main__":
    main()
