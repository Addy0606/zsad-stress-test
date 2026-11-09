# Zero-Shot Anomaly Detection Stress Test

Benchmarks three families of zero-shot anomaly detectors (CLIP-based, vision-foundation-model-based,
language-free) on industrial (MVTec AD, VisA, MPDD) and medical (BMAD Brain MRI, Head CT) images,
analyses where each fails, and trains a small fusion head on their outputs.

> **Status: work in progress.** Phase 1 (data loaders) done. No results yet. Every number in
> this repo will come from code that ran here. See [`docs/brief.md`](docs/brief.md) for the plan and
> [`PROGRESS.md`](PROGRESS.md) for the log.

## Setup (local)

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir -r requirements.txt
python scripts/setup_third_party.py                  # clones method repos at pinned commits
python scripts/download_dev_subset.py                # ~70 MB dev subset from Kaggle (needs ~/.kaggle/kaggle.json)
python -m pytest                                     # loader tests
python scripts/visualize_dataset.py --dataset mvtec  # sample grid -> outputs/samples/mvtec.png
```

Full datasets are used on Kaggle: see [`notebooks/kaggle_setup.ipynb`](notebooks/kaggle_setup.ipynb).

## Datasets

| Dataset | Domain | Test images | Pixel masks | Licence |
|---|---|---|---|---|
| MVTec AD | industrial | 1,725 | yes | CC BY-NC-SA 4.0 |
| VisA | industrial | 2,162 | yes | CC BY 4.0 |
| MPDD | industrial (test-only stress set) | 458 | yes | CC BY-NC-SA 4.0 |
| BMAD Brain MRI (BraTS 2021 slices) | medical | 3,715 | yes | CC BY-NC-SA 4.0 (BMAD); BraTS terms |
| Head CT (hemorrhage) | medical | 200 | no (image labels only) | CC0 |

## Credits

Method code is not copied into this repo; it is cloned into `third_party/` at pinned commits:
[AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) (MIT),
[AnomalyVFM](https://github.com/MaticFuc/AnomalyVFM) (no licence file),
[VisualAD](https://github.com/7HHHHH/VisualAD) (no licence file).
Datasets: MVTec AD (Bergmann et al.), VisA (Zou et al.), MPDD (Jezek et al.), BMAD (Bao et al.) / BraTS 2021,
Head CT (Kitamura, Kaggle).
