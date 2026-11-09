# PROGRESS

Running log: what was done in each phase, decisions, deviations from the spec, and open issues.

## Phase 0: Scope and method selection (started 2026-11-09)

### Done
- Answered the spec's open questions with Addy:
  - **GPU:** local RTX 3050 Laptop, 6 GB, for development; Kaggle T4 for full runs.
  - **Timeline:** compressed to about 3 days of work, with no corners cut; per-phase review gates kept.
  - **Medical:** BMAD Brain MRI (pixel masks) + Head CT (image-level only).
  - **GitHub:** `Addy0606/zsad-stress-test`, public from day one (created, empty until Phase 0 is approved).
- Checked every candidate repo: exists, weights downloadable, licence, protocol, published numbers. See `docs/brief.md` §3.
- Smoke-tested on the RTX 3050 with `scripts/phase0_smoke_test.py`:
  - AnomalyVFM-DINOv2: ✅ runs, peak 1,894 MB GPU, 96×96 output map.
  - AnomalyVFM-RADIO: ✅ peak 1,955 MB, 96×96 map.
  - AnomalyCLIP (VisA-trained checkpoint): ✅ peak 2,073 MB, 518×518 map.
  - VisualAD (VisA-trained checkpoint): ✅ peak 1,995 MB, 518×518 map.
  - All four fit easily on the 6 GB laptop GPU at batch size 1. Extra pip packages the repos need but don't list
    properly: `thop`, `tabulate`, `seaborn`, `pandas`, `scikit-image`.
  - Score scales differ a lot (AnomalyCLIP gives a probability in [0, 1]; VisualAD a raw map with values in roughly
    [-4, 3]), so the fusion head must normalise each map per image, as the spec already says.
- Disk: env 7.6 GB, weight cache 4.6 GB, third_party 1 GB (all on D:). pip's 5.2 GB download cache on C: was purged;
  installs now use `--no-cache-dir`. Datasets stay on Kaggle; only a small dev subset will be stored locally.
- Wrote `docs/brief.md`.

### Decisions
- CLIP family = AnomalyCLIP; foundation-model family = AnomalyVFM (DINOv2, plus RADIO as an extra row, pending approval); language-free = VisualAD.
  Baselines: B1 plain CLIP prompt score (single image), B2 DINOv2 mutual-NN (transductive, reported separately).
- Third-party repos are cloned into gitignored `third_party/` and never vendored (two of them have no licence).
- Environment: Python 3.12 venv at `D:\Projects\envs\zsad`; `torch==2.8.0+cu128`, `torchvision==0.23.0`.
  AnomalyVFM needs `huggingface_hub==0.36.2` and `transformers==4.56.1`. With huggingface_hub 2.x its
  `from_pretrained` drops the config and fails with a shape mismatch.

- **2026-11-09: brief approved.** AnomalyVFM-RADIO added as an extra row; VisualAD on VisA will be trained on Kaggle
  (fallback N/A); weights from the no-licence repos are downloaded at runtime and never redistributed.

### Deviations from spec
- **D1. No custom photo dataset** (Addy's decision, 2026-11-09). It is replaced by **MPDD**, a public metal-parts
  dataset with varied lighting, pose and background and pixel masks. MPDD takes the custom set's role: test-only,
  never tuned on, never used to train the fusion head. Phase 3 (shooting guide, annotation, DATACARD) is dropped.
  Research question 2 now reads "industrial → harder industrial (MPDD) and → medical".

### Open issues
- VisualAD's MVTec-trained checkpoint isn't released, so VisualAD on VisA needs our own training run (brief §8.2).
- AnomalyVFM runs in bf16, which the T4 doesn't support natively. Phase 2 will check fp16/fp32 against bf16.
- AnomalyCLIP checkpoint folder names (`9_12_4_multiscale` vs `9_12_4_multiscale_visa`) need confirming by reproduction.
- VisualAD's "Brain_AD" is assumed to be the BMAD brain split; to confirm.
- pip once silently replaced CUDA torch with a CPU build when installing other packages. The Phase 1 requirements need a pinned torch and a constraints file.

## Phase 1: Environment and data (2026-11-09)

### Done
- Repo skeleton (`configs/ data/ methods/ eval/ fusion/ analysis/ app/ notebooks/ tests/ scripts/`), `pyproject.toml` (ruff, pytest).
- `requirements.txt`: pinned, tested versions; **torch deliberately excluded** (installed separately, so pip can't swap in a CPU build).
- `scripts/setup_third_party.py`: clones AnomalyCLIP / VisualAD / AnomalyVFM at pinned commits and downloads the VisualAD LFS checkpoint (size-checked).
- Loaders behind one interface (`data/base.py`); each subclass only lists files (`_scan()`), and loading/resizing is shared:
  `mvtec.py` (MVTec AD + MPDD, same layout), `visa.py` (official `1cls.csv` split + `image_anno.csv` defect names),
  `medical_brain.py` (BMAD Brain), `medical_headct.py` (Head CT, `has_pixel_labels = False`).
  Extra item key `dataset` (needed for leave-one-dataset-out).
- Configs: `configs/datasets/<name>.yaml` (what) + `configs/paths/{local,kaggle}.yaml` (where); `data/registry.py` builds any dataset by name.
- `scripts/visualize_dataset.py` (acceptance command), `scripts/check_paths.py` (verifies roots, finds Kaggle mounts),
  `scripts/download_dev_subset.py` (deterministic 67 MB local subset), `notebooks/kaggle_setup.ipynb` (thin wrapper).
- Tests: 30 pass: interface keys and types, masks binary, normal → empty mask, anomalous → non-empty mask (when the dataset has masks),
  only the test split is loaded, resize keeps masks binary, VisA multi-defect labels, and the same checks on the real dev subset.

### Verified dataset facts (Kaggle copies, from file listings; nothing downloaded in full)
| Dataset | Kaggle ref | Test images | Matches official? |
|---|---|---|---|
| MVTec AD | `ipythonx/mvtec-ad` | 1,725 | yes (official 1,725) |
| VisA | `tensura3607/amazon-visa-anomaly` | 2,162 (962 normal / 1,200 anomalous) | yes (official 1cls split) |
| MPDD | `alirezasalehy/mpdd-ad` | 458 | yes, MVTec layout |
| BMAD Brain | `duongkaiba/bmad-aaclip` (`Brain_AD/BraTS2021_slice`) | 3,715 (640 normal / 3,075 tumour) | BMAD layout; masks in `test/Ungood/label/` |
| Head CT | `felipekitamura/head-ct-hemorrhage` | 200 (100 / 100) | original upload; licence **CC0** |

### Decisions
- Loaders return images at **native resolution** by default; resizing is per method (fixed in Phase 2 configs).
- Head CT keeps the uniform interface (all-zero masks) and sets `has_pixel_labels = False` so evaluation skips pixel metrics.
- BMAD `valid` split and all `train` splits are never loaded.

### Open issues / observations
- Some Head CT images carry text overlays (panel letters like "B", "NCCT"), apparently taken from publications.
  Possible shortcut cue; note for Phase 5.
- `configs/paths/kaggle.yaml` roots are unverified until the setup notebook runs on Kaggle.
- The pinned `requirements.txt` hasn't been tested inside the Kaggle image yet (it may conflict with preinstalled packages).
