"""Save the pixel-fusion normaliser the demo uses (Phase 7), and check it reproduces Phase 6.

The demo fuses the four detectors' maps of a new photo with the "mean of normalised maps" head. For a new photo,
the right normaliser is the one fitted on all three training datasets (MVTec AD, VisA, BMAD Brain): the one
Phase 6 used for the test-only datasets MPDD and Head CT. pixel_fusion.py didn't save it, so this script
re-creates it with the same code, data order and seeds, then checks it: re-fusing MPDD images must give the
image scores pixel_fusion.py stored for them.

Runs as a Kaggle CPU job with the same inputs as pixel_fusion.py:
    python fusion/export_normalizer.py --out /kaggle/working/normalizer
Output: fusion_normalizer.npz with grid (1001, 4) and the detector order. It is tiny and goes in app/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fusion.pixel_fusion import DETECTORS, SEED, DatasetMaps, fit_normalizer, sample_pixels  # noqa: E402
from fusion.splits import FOLD_DATASETS, outer_splits  # noqa: E402
from methods.base import topk_mean  # noqa: E402

N_CHECK = 40


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/normalizer")
    p.add_argument("--paths", default="kaggle")
    a = p.parse_args()

    manifest = OmegaConf.load(REPO / "configs" / "bench_sources.yaml")
    rng = np.random.default_rng(SEED)  # same generator, same order as pixel_fusion.main()
    samples = {}
    for name in FOLD_DATASETS:
        print(f"===== sampling {name}", flush=True)
        samples[name] = sample_pixels(DatasetMaps(name, a.paths, Path(a.input), manifest), rng)
    split = next(s for s in outer_splits() if s.test == "mpdd")
    norm = fit_normalizer([samples[t] for t in split.train], np.random.default_rng(SEED))  # as in run_split()

    # check: re-fuse some MPDD images and compare with the image scores Phase 6 stored
    mpdd = DatasetMaps("mpdd", a.paths, Path(a.input), manifest)
    stored_csv = next(Path(a.input).rglob("pixel_fusion/mpdd/mean/image_scores.csv"))  # Phase 6 job output
    with open(stored_csv, newline="", encoding="utf-8") as f:
        stored = {r["image"]: float(r["score"]) for r in csv.DictReader(f)}
    idx = np.random.default_rng(1).choice(len(mpdd.keys), N_CHECK, replace=False)
    diffs = []
    for i in idx:
        m = mpdd.maps(int(i))
        fused = norm.transform(m.reshape(4, -1).T).mean(1).reshape(m.shape[1:]).astype(np.float32)
        diffs.append(abs(topk_mean(fused, 0.01) - stored[mpdd.keys[i]]))
    print(f"max |re-fused - stored| image score over {N_CHECK} MPDD images: {max(diffs):.2e}", flush=True)
    if max(diffs) > 1e-5:
        raise SystemExit("normaliser does not reproduce Phase 6: not saving it")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "fusion_normalizer.npz", grid=norm.grid_, detectors=np.array(DETECTORS), train=np.array(split.train))
    print("saved", out / "fusion_normalizer.npz", "train:", split.train)


if __name__ == "__main__":
    main()
