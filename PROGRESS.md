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

## Phase 2: Evaluation harness and reproduction (2026-11-09 – 2026-11-10)

### Done
- `eval/metrics.py`: image AUROC, image F1-max, pixel AUROC, exact AUPRO (FPR ≤ 0.3, normalised by 0.3).
  Per-category values, then the mean over categories. 18 tests, including brute-force AUPRO checks.
- `eval/cache.py` + `eval/run_benchmark.py`: per-image resumable cache (`scores.csv` + float16 256×256 maps).
  Each run also writes `config.yaml` (resolved config, git commit, package versions, GPU) and `run_info.json` (s/img, peak GPU MB).
  `eval/evaluate.py` turns a cache into `metrics.json`.
- Method wrappers behind one interface (`methods/base.py`): AnomalyCLIP, VisualAD, AnomalyVFM (DINOv2 and RADIO),
  B1 CLIP prompt, B2 DINOv2 mutual nearest neighbour (transductive).
- Kaggle pipeline: `scripts/kaggle_submit.py` (fills the notebook parameters, attaches only the needed datasets),
  `kaggle_wait.py`, `kaggle_fetch.py` (downloads everything except maps).

### Reproduction (image AUROC / pixel AUROC / AUPRO, mean over categories; ours from `outputs/kaggle/zsad-repro-*/…/metrics.json`)
| Method | Dataset | Ours | Published | Image AUROC Δ |
|---|---|---|---|---|
| AnomalyCLIP (layer 24, official setting) | MVTec | 91.6 / 91.1 / 85.1 | 91.5 / 91.1 / 81.4 | +0.1 ✅ |
| AnomalyCLIP (layer 24, official setting) | VisA | 82.0 / 95.5 / 86.9 | 82.1 / 95.5 / 87.0 | −0.1 ✅ |
| AnomalyVFM (DINOv2) | MVTec | 87.4 / 91.4 / 85.3 | 87.4 / 91.6 / 84.5 | 0.0 ✅ |
| AnomalyVFM (DINOv2) | VisA | 89.3 / 94.2 / 82.8 | 89.3 / 94.5 / 84.5 | 0.0 ✅ |
| AnomalyVFM (RADIO) | MVTec | 96.4 / 91.3 / 87.2 | 96.4 / 92.0 / 87.6 | 0.0 ✅ |
| AnomalyVFM (RADIO) | VisA | 92.9 / 94.9 / 88.3 | 92.9 / 95.8 / 90.5 | 0.0 ✅ |
| VisualAD (authors' VisA-trained ckpt) | MVTec | 92.2 / 90.8 / 87.7 | 92.2 / 90.8 / 87.5 | 0.0 ✅ |
| VisualAD (our MVTec-trained ckpt, batch 4) | VisA | 83.1 / 95.3 / 89.8 | 84.7 / 95.8 / 91.0 | −1.6 ✅ |

Baselines (no published target; sanity checks). Image AUROC / F1-max / pixel AUROC / AUPRO:
| Baseline | MVTec | VisA |
|---|---|---|
| B1 CLIP prompt (single image) | 86.6 / 89.2 / 40.8 / 8.0 | 77.1 / 78.8 / 52.1 / 15.5 |
| B2 DINOv2 mutual NN (**transductive**, reported separately) | 92.5 / 93.4 / 94.9 / 90.6 | 82.7 / 81.3 / 95.6 / 83.0 |

- B1's pixel AUROC is at or below chance: raw CLIP patch tokens don't align with text (the known failure that AnomalyCLIP/VisualAD address).
  Image-level is reasonable because the score uses the pooled embedding and a prompt ensemble.
- B2 is strong at pixel level but uses the other unlabelled test images of the category, so it is not comparable to single-image methods.

- AnomalyCLIP checkpoint pairing confirmed: VisA-trained prompts for MVTec, MVTec-trained prompts for VisA (as in the official `test.sh`).
- Pixel-metric gaps, investigated one hypothesis at a time (first AnomalyCLIP run used layers [6, 12, 18, 24]: 88.2 / 81.9 MVTec, 93.9 / 83.5 VisA):
  - H1, **rejected**: evaluation resolution. `--eval-size 518` (AnomalyCLIP's input size, as in its `test.py`) gives the same numbers
    (MVTec 88.2 / 82.1, VisA 93.9 / 82.9; `outputs/kaggle/zsad-repro-evalsize`). 256 stays the common evaluation size.
  - H2, **confirmed**: AnomalyCLIP layers. The official `test.sh` passes `--features_list 24`; the `test.py` default is [6, 12, 18, 24].
    With layer 24, pixel AUROC matches exactly (91.1, 95.5) and VisA AUPRO is within 0.1 (`zsad-repro-anomalyclip-l24`).
    `configs/methods/anomalyclip.yaml` now uses [24].
  - H3, **confirmed**: AUPRO implementation. AnomalyCLIP's `cal_pro_score` (200 linear thresholds, keeps FPR < 0.3, then min-max
    rescales the FPR axis) applied to *our* layer-24 maps gives MVTec 81.2 (published 81.4) and VisA 86.0 (published 87.0); our exact
    AUPRO gives 85.1 / 86.9. Their function scores lower on 15/15 MVTec categories (up to 14 points: grid 89.6 vs 75.6)
    (`scripts/compare_aupro.py`, `outputs/kaggle/zsad-aupro-compare`).
    We keep the exact, standard AUPRO (tested against brute force); **published AUPRO values from different papers aren't directly comparable**.
  - AnomalyVFM VisA AUPRO is 1.7 (DINOv2) / 2.2 (RADIO) below published. Likely the same kind of cause: AnomalyVFM scores with the
    `adeval` library (binned accumulation; masks resized by nearest-neighbour to the model's map size). **Not verified**:
    we haven't run `adeval` on our maps. Image AUROC and pixel AUROC are within 0.9.
- Cost on Kaggle T4 (from `run_info.json`): AnomalyCLIP 0.64–0.66 s/img, 2.05 GB peak;
  AnomalyVFM DINOv2 0.21–0.23 s/img, 1.88 GB; RADIO 0.23–0.24 s/img, 1.94 GB.
  (VisualAD and baselines: see their `run_info.json`; full cost table comes in Phase 4.)
- Local RTX 3050 runs hit intermittent CUDA OOM under Windows with >3 GB reported free; all GPU work now runs on Kaggle.

### Deviations
- **D2: VisualAD MVTec checkpoint trained with batch size 4, not 8.** Batch 8 at 518 px ran out of memory on the 16 GB T4
  (14.5 GB in use at the first step; the authors' `train.py` has AMP hard-coded off and defaults to `cuda:1`).
  Everything else follows their `CLIP.sh` (2 epochs, lr 1e-3, seed 111). The check is the MVTec→VisA image AUROC vs the published 84.7.
- VisualAD training runs with **Pillow 11.3** (training subprocess only, via `PYTHONPATH`): the authors' `dataset.py` calls
  `Image.fromarray(float64, mode="L")`, which Pillow 13 rejects. Verified locally: 11.3 gives the intended all-zero mask, 13 raises. Evaluation is unaffected.

### Phase 2 acceptance
- Image AUROC within 2 points of published for every method/dataset: **yes** (largest gap: VisualAD on VisA, −1.6, explained by D2 at most).
- Every gap > 2 points explained: AnomalyCLIP pixel metrics (layers; AUPRO implementation), both verified;
  AnomalyVFM RADIO VisA AUPRO (−2.2), likely implementation, unverified.
- Tests: 51 pass (loaders, metrics, end-to-end evaluation incl. `--eval-size`).

### Problems hit and fixed
- Kaggle mounted datasets at `/kaggle/input/datasets/<owner>/<slug>` in some sessions and `/kaggle/input/<slug>` in others.
  The notebook now runs `check_paths.py --fix` first, which finds each dataset by marker file and rewrites `configs/paths/kaggle.yaml`.
- Kaggle allows only 2 concurrent GPU batch sessions; a third push is refused (`kaggle_submit.py` now exits on that error).
- `kaggle_fetch.py`: logs read as UTF-8; retries on HTTP 429.
- A Kaggle submission sat in "QUEUED" for 4 h with nothing in the queue; re-pushing started it immediately.
  Rule: if a job is queued > 30 min, check the site and re-push.
- `kaggle_submit.py` gained `--cpu`, `--datasets` and `--kernel-sources` (attach an earlier job's outputs) for post-processing jobs.
