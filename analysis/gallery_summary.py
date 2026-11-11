"""What kind of defects fill the failure galleries (analysis/galleries.py)?

Counts how often the 10 worst missed anomalies per detector and industrial dataset are *logical* anomalies (a part
missing, misplaced, flipped, swapped, extra, of the wrong type, or mismatched: each surface looks fine, but the
object is wrong), and compares that with how common such defects are among all anomalous test images.
Also counts how many images appear in more than one detector's worst-miss list.

Reads analysis/galleries/galleries.csv and the benchmark's scores.csv files (cached outputs); writes
analysis/results/gallery_summary.csv.

    python analysis/gallery_summary.py
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from bench_stats import find_run  # noqa: E402

INDUSTRIAL = ("mvtec", "visa", "mpdd")
# defect types (dataset folder names) where the object's parts are wrong rather than its surface damaged
LOGICAL = {
    "missing",  # VisA
    "extra",  # VisA
    "misplaced",  # MVTec AD transistor
    "flip",  # MVTec AD metal nut
    "pill_type",  # MVTec AD pill
    "missing_cable",  # MVTec AD cable
    "cable_swap",  # MVTec AD cable
    "parts_mismatch",  # MPDD
    "bend_and_parts_mismatch",  # MPDD
}


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    gallery = read(REPO / "analysis" / "galleries" / "galleries.csv")
    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    rows = []
    shares = []
    for ds in INDUSTRIAL:
        src = manifest.runs.anomalyclip[ds]  # any run: every method scores the same test images
        anomalous = [
            r
            for r in read(find_run(REPO / "outputs" / "kaggle", src.kernel, src.run) / "scores.csv")
            if r["label"] == "1"
        ]
        base = sum(r["defect_type"] in LOGICAL for r in anomalous) / len(anomalous)
        missed = [r for r in gallery if r["dataset"] == ds and r["kind"] == "missed"]
        share = sum(r["defect_type"] in LOGICAL for r in missed) / len(missed)
        counts = collections.Counter(r["image"] for r in missed)
        shares.append((len(missed), share, len(anomalous), base))
        rows.append(
            {
                "dataset": ds,
                "worst_miss_entries": len(missed),
                "logical_share_of_worst_misses": round(share, 4),
                "anomalous_test_images": len(anomalous),
                "logical_share_of_all_anomalous": round(base, 4),
                "distinct_images_in_worst_miss_lists": len(counts),
                "images_in_2plus_lists": sum(v >= 2 for v in counts.values()),
            }
        )
    n_m = sum(s[0] for s in shares)
    n_a = sum(s[2] for s in shares)
    rows.append(
        {
            "dataset": "all industrial",
            "worst_miss_entries": n_m,
            "logical_share_of_worst_misses": round(sum(s[0] * s[1] for s in shares) / n_m, 4),
            "anomalous_test_images": n_a,
            "logical_share_of_all_anomalous": round(sum(s[2] * s[3] for s in shares) / n_a, 4),
            "distinct_images_in_worst_miss_lists": sum(r["distinct_images_in_worst_miss_lists"] for r in rows),
            "images_in_2plus_lists": sum(r["images_in_2plus_lists"] for r in rows),
        }
    )
    out = REPO / "analysis" / "results" / "gallery_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
