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

## Phase 4: Full benchmark (2026-11-10)

### Done
- All 30 runs: 6 methods × 5 datasets (MVTec AD, VisA, MPDD, BMAD Brain, Head CT), on Kaggle T4s.
  `configs/bench_sources.yaml` records the Kaggle job and run folder behind every table cell.
- `eval/pixel_stats.py`: per-image histograms (normal pixels, defect pixels, each defect region) on shared quantile bins,
  so pixel AUROC and AUPRO can be bootstrapped without the 3–5 GB of maps. Computed on Kaggle by `scripts/bench_stats.py`
  (CPU job). **Binned vs exact, worst case over all 24 pixel-labelled runs: 0.025 points.**
- `eval/bootstrap.py`: 95% percentile bootstrap, 1,000 resamples, images resampled within each category and label;
  all four metrics from the same replicates. Headline numbers are the exact metrics; bins are only used for CIs.
- `eval/make_tables.py`: **one command** (`python eval/make_tables.py`, ~13 min) writes `results/benchmark.md` and `.csv`
  with every metric, its CI, image count and source run, plus the cost table.
- Tests: 60 pass (adds bootstrap, binned metrics, cross-root matching, F1 floor).

### Published numbers on the new datasets (image AUROC / pixel AUROC / AUPRO)
| Method | Dataset | Ours | Published |
|---|---|---|---|
| VisualAD | BMAD Brain | 80.8 / 95.2 / 79.9 | 80.8 / 95.2 / 79.5 ✅ |
| AnomalyVFM DINOv2 | MPDD | 83.4 / 96.1 / 88.1 | 83.4 / 96.2 / 87.4 ✅ |
| AnomalyVFM RADIO | MPDD | 85.7 / 95.9 / 88.4 | 85.7 / 96.8 / 90.2 ✅ |
| AnomalyVFM DINOv2 / RADIO | Head CT | 88.2 / 94.0 (image only) | 88.2 / 94.0 ✅ |
| AnomalyCLIP | Head CT | 93.1 (image only) | 93.4 ✅ |
| AnomalyCLIP | BMAD Brain | 60.7 / 96.5 / 79.7 | 69.0 / 95.1 / 71.5 ⚠ |

- AnomalyCLIP on BMAD Brain is 8.3 points below the 69.0 we compared against. That number is from **VisualAD's** paper
  (its own run of AnomalyCLIP, settings not stated), not from AnomalyCLIP's authors, and the brief already marked the
  dataset match as an assumption. Our pixel AUROC and AUPRO are *higher* than theirs, so it isn't a broken run.
  **Unexplained; carried to Phase 5.**

### Observations to test in Phase 5 (not yet findings)
- AnomalyVFM RADIO has the highest image AUROC of the single-image methods on all five datasets
  (CIs overlap on Head CT).
- Every method loses image AUROC from MVTec to MPDD (e.g. VisualAD 92.2 → 71.5).
- AnomalyCLIP and VisualAD share the same frozen CLIP backbone but differ by 20 image-AUROC points on BMAD Brain (60.7 vs 80.8).
- AnomalyVFM DINOv2 has low AUPRO on Brain (60.8) despite 93.3 pixel AUROC: small tumour regions are missed.
- B2 (transductive) is strong on MVTec (92.5) but drops to 64.8 on MPDD and 61.1 on Head CT, where test images vary
  more in pose and appearance, so "no close match in other images" stops meaning "defect".
- B1 pixel AUROC is at or below 50 on every dataset (CLIP patch tokens don't align with text).

### Metric caveats (written into results/benchmark.md)
- **F1-max floor.** Flagging every image anomalous scores F1 = 2p/(1+p). On BMAD Brain (83% anomalous) that is 90.6,
  which is why every method, B1 included, scores ~90.6–92.3 there. The table shows the floor per dataset.
- **F1-max CIs lean upward.** F1-max maximises over thresholds; resamples with repeated images fit thresholds more
  easily, so the percentile bootstrap overestimates. Image AUROC is the comparison metric.
- Head CT has image labels only (no pixel metrics).

### Cost (T4, inference only, image-weighted over all 8,260 test images)
AnomalyCLIP 0.554 s/img, 2.0 GB · VisualAD 0.386 s, 1.9 GB · AnomalyVFM DINOv2 0.202 s, 1.9 GB ·
RADIO 0.216 s, 1.9 GB · B1 0.147 s, 1.7 GB · B2 0.107 s (+ per-category reference prep, not timed), 2.6 GB.

### Problems hit and fixed
- Scores stored absolute image paths, but Kaggle mounted the same dataset at different folders in different sessions.
  Images are now matched by path relative to the dataset root recorded in each run's `config.yaml` (test added).
- `kaggle_fetch.py` listed outputs 20 files per page, so jobs with ~13k maps hit HTTP 429 and fetches were silently cut
  short (B2 looked as if it had crashed; it hadn't). Now 500 per page.
- GPU jobs waited ~80 min in Kaggle's queue; `kaggle_wait.py --max-queue-min` now flags long waits.
  (Re-pushing was refused while queued: the queue was real this time.)
- Git history rewritten so every commit is authored by Aadhitya Ayyappan with no co-author trailers (user request).

### Phase 4 acceptance
- Complete method × dataset × metric table with 95% CIs: **yes** (30 runs; Head CT image-only; B2 separate).
- Runtime and peak memory per method on one GPU type: **yes**.
- Regenerated from cache with one command: **yes** (`python eval/make_tables.py`).
- Open: the cached inputs live in `outputs/` (gitignored); publishing them for a fresh clone is a Phase 8 task.

## Phase 5: Failure analysis (2026-11-10 – 2026-11-11)

### Done
- `analysis/FINDINGS.md`: **five findings**, each with a table, a figure and 95% bootstrap CIs.
  1. AnomalyCLIP's *map* finds brain tumours its image score misses: scoring Brain MRI images from the map's top 1%
     gives 85.1 vs 60.7 image AUROC (+24.4 [+22.2, +26.6], paired). For AnomalyVFM the reverse: its head beats its map.
  2. Right image, wrong place: among correctly flagged anomalies, RADIO's map peak is off the defect in 36–54% of
     industrial images; VisualAD 11–17%.
  3. Methods miss different images (oracle roughly halves misses), but an OR rule held at 5% total FPR only helps on
     MPDD. Sets the bar for Phase 6.
  4. Tiny defects (AnomalyVFM DINOv2 finds 19% of tiny MPDD regions vs 61–66% for CLIP-backbone methods) and
     missing/mismatched parts (every method's CI includes chance) are blind spots.
  5. Domain shift: VisualAD loses 16 points on MPDD but ~2 (n.s.) on medical; AnomalyCLIP loses ~10 on medical.
- Code: `analysis/regions.py` and `analysis/map_scores.py` (Kaggle CPU jobs over the cached maps),
  `analysis/image_level.py`, `analysis/region_level.py`, `analysis/figures.py` (local, from small CSVs).
  Results in `analysis/results/*.csv`, figures in `analysis/figures/`. Tests: 72 pass.

### Checks and corrections made along the way
- The first "oracle" OR-ed four detectors each at 5% FPR, which flags 11–15% of normal images. Added the label-aware
  oracle (clearly marked as an upper bound) and an OR rule held at 5% total FPR (achievable).
- Bootstrap CIs of 5%-FPR miss rates initially didn't contain their own point estimates: a 5% threshold on 20–60
  normal images is an extreme order statistic and re-estimating it on resamples biases it. Thresholds now use an
  actual normal score (at most 5% exceed it) and are held fixed; CIs cover image sampling. All CIs bracket the estimate (checked).
- Every number in FINDINGS.md was re-checked against the CSVs; five wrong statements in the first draft were corrected.
- AnomalyCLIP on Brain MRI with the 4-layer setting gives the same image AUROC (60.7): the image score doesn't use the
  map. VisualAD's 69.0 is not reproduced by any variant (own score, map max, top-1%, top-5%). Left open.
- VisualAD's map-top-1% score equals its own score on industrial data (difference 0.0), confirming the map-score pipeline.

### Open (not checked)
- Head CT text-overlay shortcut (no labels for which images carry text).
- "Large regions found less often" is partly an artefact of the 50%-overlap definition; reported with that caveat.

### Phase 5 acceptance
- 3–5 findings, each backed by a plot or table with a CI: **yes** (5).
- Every number generated by scripts from cached outputs: **yes** (sources named per finding).

## Phase 6: Fusion with leave-one-dataset-out (2026-11-11)

### Done
- Splits (`fusion/splits.py`): leave-one-dataset-out over MVTec AD, VisA and BMAD Brain; MPDD and Head CT are
  test-only, scored by heads trained on all three. Hyperparameters chosen by an inner leave-one-dataset-out on the
  training datasets only. Tests assert no held-out dataset reaches training or validation (`tests/test_fusion.py`).
- Normalisation (`fusion/normalize.py`): per-feature quantile transform fitted on training data only.
- Image-level heads (`fusion/image_fusion.py`, local): best single, mean of own scores, mean of 16 features
  (own score + map max/top-1%/top-5% per detector), logistic regression, boosted trees; oracle selection for reference.
- Pixel-level heads (`fusion/pixel_fusion.py`, Kaggle CPU job `zsad-p6-pixelfusion`): best single map (by training
  AUPRO), mean of normalised maps, logistic regression on the four normalised map values; trained on sampled pixels
  (50 normal + 50 defect per image, datasets weighted equally). Exact metrics on every test pixel.
- Report (`fusion/report.py`): bootstrap CIs and paired differences; all tables in `results/fusion.md`.

### Results (held-out data only; full tables with CIs in results/fusion.md)
- **Image level: negative result.** RADIO alone (chosen on training data, also best in hindsight) is not beaten on
  the industrial sets: logistic regression −1.9 [−2.8, −1.1] on MVTec AD, −4.0 [−5.0, −2.9] on VisA, −2.7 (n.s.)
  on MPDD; +0.3 (n.s.) on Brain MRI. Only Head CT gains (+2.8 [+0.5, +5.7]; misses at 5% FPR 28% → 9%, equal to the
  Phase 5 oracle). Boosted trees are the worst learned head everywhere. Brain MRI misses drop 56.0% → 47.9%
  (oracle 44.5%), but industrial misses rise.
- **Pixel level: positive result.** Averaging the four normalised maps beats the best single map (VisualAD) on all
  four held-out sets with masks: AUPRO +2.3 [+1.7, +2.9] MVTec AD, +1.5 [+1.1, +1.9] VisA, +1.8 [+1.1, +2.5] Brain,
  +1.1 [+0.6, +1.5] MPDD (never trained on); pixel AUROC +0.6 to +2.4, all CIs above zero.
  Logistic regression learns near-equal weights (0.89–1.01) and matches the plain mean, so learning adds nothing.
- Scoring images from the fused map (top 1%) is worse than RADIO's own score (−3.1 to −8.5 on industrial data).

### Caveats
- The pixel baseline is the best single map by AUPRO. On pixel AUROC another detector is sometimes better in
  hindsight (benchmark.csv: AnomalyVFM DINOv2 91.4 on MVTec AD, AnomalyCLIP 95.5 on VisA and 96.5 on Brain);
  the fused map (93.0 / 96.0 / 97.6) still beats those.
- The transductive B2 baseline (DINOv2 mutual NN, not a fusion input) still beats the fused map on MVTec AD
  (AUPRO 90.6 vs 90.0, pixel AUROC 94.9 vs 93.0).
- Only three training datasets: each outer fold trains on two, so the inner selection runs on one-vs-one splits.

### Phase 6 acceptance
- Fusion evaluated on held-out datasets only, against the best single method and the oracle: **yes**.
- Split leakage test: **yes** (79 tests pass, ruff clean).

## Phase 7: Demo app (2026-11-11)

### Done
- **Live demo: https://huggingface.co/spaces/addy0606/zsad** (Gradio 5.50, ZeroGPU, public).
- `app/app.py`: upload a photo, pick industrial/medical; shows the fused map and all four detectors' maps over
  the photo, coloured on one common scale (each map through the Phase 6 training-data normaliser), plus each
  method's raw image score, its percentile among normal reference images, and the time it took.
  Optional B1 baseline, the only method that reads the category name (the four detectors ignore it by design).
- `fusion/export_normalizer.py` (Kaggle CPU job `zsad-p7-normalizer`): re-creates the normaliser Phase 6 used for
  MPDD/Head CT (fitted on MVTec AD + VisA + BMAD Brain) and checks it: re-fused scores of 40 MPDD images differ
  from the stored Phase 6 scores by 0.0. Saved as `app/fusion_normalizer.npz` (32 KB).
- `app/build_reference.py`: normal-image reference scores (`app/score_reference.json`): MPDD normals for industrial,
  Brain MRI + Head CT normals for medical; same checkpoints as the demo.
- `scripts/deploy_space.py`: builds the Space from repo files only and uploads it. Third-party code and weights are
  downloaded from the authors at startup (Phase 0 licence decision).

### Checks
- Same screw image (MVTec, scratch_head/003) locally and on the Space: AnomalyCLIP 0.986 / 0.9856,
  VisualAD 0.637 / 0.6370, AnomalyVFM DINOv2 0.500 / 0.5000, RADIO 0.554 / 0.5532, B1 0.899 / 0.8992.
- Time per photo through the public API (from this laptop, Space awake): 18.5, 12.2, 13.6 and 9.5 s wall clock;
  model compute 0.6–1.3 s, the rest is ZeroGPU attaching a GPU and image transfer. Under the 30 s target.
- On CPU (2 threads, local) the four detectors took ~45 s together, so CPU hosting would have missed the target.

### Deviations and notes
- Hosting is ZeroGPU, not CPU: Hugging Face refuses Gradio Spaces on the free CPU tier without PRO; a ZeroGPU Space
  created on the website is free (Addy's account).
- Gradio pinned to 5.50: Gradio 6 needs huggingface_hub >= 1.16, which breaks AnomalyVFM's loader (needs 0.36.2).
- A clean environment exposed seven packages the third-party repos import without declaring
  (setuptools for pkg_resources, thop, tabulate, scikit-learn, scikit-image, opencv, matplotlib/seaborn); pinned in
  `app/requirements.txt`.
- Locally on Windows with the CPU-only PyTorch build, AnomalyVFM DINOv2 crashed while loading its weights (access
  violation; ~4.8 GB RAM free). It loads fine on the Space (Linux). Not investigated further.
- Cold start: after 48 h without visitors the Space sleeps; the first visitor then waits ~2–4 min for it to start.
- The fused map's top-1% image score is shown but labelled as weak (Phase 6); the screw example sits at the 38th
  percentile of normal MPDD images while every detector puts it at or above the 95th.

### Phase 7 acceptance
- Public Spaces link that processes a new photo in under 30 s: **yes on benchmark images (9.5–18.5 s)**; a photo
  from outside the datasets still to be tried by Addy.
