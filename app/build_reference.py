"""Reference scores of NORMAL images, so the demo can say how unusual a photo's score is (Phase 7).

Each method's raw image score lives on its own scale (AnomalyCLIP gives ~0.98 for almost everything, VisualAD
gives negative numbers on MPDD), so the demo reports "higher than X% of normal reference images" instead.

References (normal test images only, scored in Phase 4/6 with exactly the checkpoints the demo uses):
- industrial: MPDD (test-only in this project, never tuned on)
- medical:    BMAD Brain MRI + Head CT for the detectors and B1; Head CT only for the fused map, because the
              demo's fusion normaliser is the one Phase 6 used for MPDD and Head CT (trained on the other three)

Reads outputs/kaggle/zsad-p5-mapscores/ and outputs/kaggle/zsad-p6-pixelfusion/, writes app/score_reference.json.

    python app/build_reference.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAP_SCORES = REPO / "outputs" / "kaggle" / "zsad-p5-mapscores" / "map_scores"
FUSION = REPO / "outputs" / "kaggle" / "zsad-p6-pixelfusion" / "pixel_fusion"
METHODS = ("anomalyclip", "visualad", "anomalyvfm_dinov2", "anomalyvfm_radio", "clip_prompt")
DOMAINS = {"industrial": ("mpdd",), "medical": ("bmad_brain", "headct")}
FUSED_DOMAINS = {"industrial": ("mpdd",), "medical": ("headct",)}


def normal_scores(path: Path) -> list[float]:
    with open(path, newline="", encoding="utf-8") as f:
        return [float(r["score"]) for r in csv.DictReader(f) if r["label"] == "0"]


def main() -> None:
    ref = {}
    for domain, datasets in DOMAINS.items():
        ref[domain] = {"datasets": list(datasets)}
        for m in METHODS:
            scores = [s for d in datasets for s in normal_scores(MAP_SCORES / m / f"{d}.csv")]
            ref[domain][m] = sorted(round(s, 6) for s in scores)
        fused = [s for d in FUSED_DOMAINS[domain] for s in normal_scores(FUSION / d / "mean" / "image_scores.csv")]
        ref[domain]["fused"] = sorted(round(s, 6) for s in fused)
        ref[domain]["fused_datasets"] = list(FUSED_DOMAINS[domain])
        print(domain, {k: len(v) for k, v in ref[domain].items() if k not in ("datasets", "fused_datasets")})
    out = Path(__file__).with_name("score_reference.json")
    out.write_text(json.dumps(ref), encoding="utf-8")
    print("wrote", out, f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
