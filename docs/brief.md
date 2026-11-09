# Project brief: Zero-Shot Anomaly Detection Stress Test

*Phase 0 deliverable. Status: **approved by Addy, 2026-11-09**. Decisions on §8: (1) run RADIO as an extra row: yes; (2) VisualAD on VisA: try training on Kaggle first, N/A as fallback; (3) runtime download of weights from the no-licence repos: accepted.*

## 1. Research questions

1. **Failure modes.** Which defect types, sizes and contrasts does each detector family miss?
2. **Domain shift.** How much does performance drop from the standard industrial benchmarks
   (MVTec AD, VisA) to (a) harder industrial images (MPDD) and (b) medical images
   (brain MRI, head CT)?
   *Changed from the spec, which used self-collected photos for (a). See PROGRESS.md, deviation D1.*
3. **Complementarity and fusion.** Do the families fail on different images? If so, does a small
   fusion head trained on their outputs beat the best single method on held-out datasets?
4. **Score–map consistency.** How often does a method flag the right image for the wrong reason,
   i.e. the image is correctly scored anomalous but the highest point of its map lies outside the
   ground-truth mask?

## 2. Closest existing work and what it leaves out

| Work | What it does | What it leaves out (our angle) |
|---|---|---|
| Method papers with wide evaluation: AnomalyCLIP (ICLR 2024), AdaCLIP (ECCV 2024), VisualAD (CVPR 2026), AnomalyVFM (CVPR 2026) | Each reports one method on 9–14 industrial and medical datasets | Dataset-level averages only. No breakdown by defect size or contrast, no cross-family complementarity, no score–map consistency check. Each paper tunes its own protocol details, e.g. VisualAD uses a top-5% pixel score on medical images and top-1% on industrial ones |
| BMAD (Bao et al., CVPR-W 2024) | Medical anomaly-detection benchmark with six datasets, three with pixel masks | Covers medical images only and benchmarks methods trained on normal images, not zero-shot families |
| Marzullo & Ranzini, *"Exploring ZSAD with CLIP in Medical Imaging: Are We There Yet?"* (2024) | Tests CLIP-based zero-shot detectors on brain tumour MRI (BraTS-MET) | CLIP-only, one medical dataset, no industrial comparison, no fusion |
| RobustAD (CVPR-W 2025) | Real-world robustness benchmark; compares zero-shot (WinCLIP, AnomalyCLIP), unsupervised and supervised detectors under shift | Industrial only. *(Summary based on the abstract; full PDF not accessible.)* |
| RADIOlogy (arXiv 2610.40580, Oct 2026) | Training-free RADIO-based detector; reports industrial and medical results and describes some of its own failure modes | Failure analysis covers only its own method, not a comparison across families |

**Gap:** I found no work that runs one CLIP-based, one foundation-model-based and one
language-free detector under a **single shared harness**, breaks their failures down by
**defect attributes**, measures their **complementarity**, and tests whether a **leave-one-dataset-out
fusion head** helps. Not finding such a work doesn't prove none exists. I'll re-check before writing the report.

## 3. Methods (one per family plus baselines)

Every candidate repo was checked on 2026-11-09. *Smoke test* means the released weights load and
one image runs end to end on an RTX 3050 6 GB laptop GPU (`scripts/phase0_smoke_test.py`).
A 6 GB card is a stricter test than a 16 GB T4.

| Family | Method | Code / weights | Licence | Protocol | Smoke test |
|---|---|---|---|---|---|
| CLIP-based | **AnomalyCLIP** (Zhou et al., ICLR 2024) | [zqhang/AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP); MVTec-trained and VisA-trained prompt checkpoints are in the repo (22 MB each) | MIT | Cross-dataset: prompts trained on MVTec AD for testing on all other datasets, on VisA for testing on MVTec AD | ✅ 2.1 GB peak, 518 px, 518×518 map |
| Vision foundation model | **AnomalyVFM** (Fučka et al., CVPR 2026), DINOv2 backbone | [MaticFuc/AnomalyVFM](https://github.com/MaticFuc/AnomalyVFM); [HF weights](https://huggingface.co/MaticFuc/anomalyvfm_dinov2) | **No licence file** in either the repo or the HF model card | Trained only on **synthetic** images generated with FLUX and inpainted, never on real benchmark images. So it's zero-shot on every dataset and one checkpoint is used everywhere | ✅ 1.9 GB peak, 672 px input, 96×96 map |
| Vision foundation model (extra row) | AnomalyVFM, RADIO backbone | [HF weights](https://huggingface.co/MaticFuc/anomalyvfm_radio) | No licence file; RADIO itself is under NVIDIA's non-commercial licence | Same as above | ✅ 2.0 GB peak, 768 px, 96×96 map |
| Language-free | **VisualAD** (Hou et al., CVPR 2026), CLIP ViT-L/14@336 *visual encoder only* | [7HHHHH/VisualAD](https://github.com/7HHHHH/VisualAD); only a **VisA-trained** checkpoint is released (126 MB, Git LFS) | **No licence file** | Learned anomaly and normal tokens trained on VisA and tested on everything else. For VisA itself it must be trained on MVTec, and that checkpoint isn't released | ✅ 2.0 GB peak, 518 px, 518×518 map |
| Baseline B1 | Plain CLIP prompt score: cosine similarity of patches with "a photo of a damaged {category}" vs "a photo of a flawless {category}" | Own code, ~50 lines, `open_clip` | — | Pure zero-shot, one image at a time | Phase 2 |
| Baseline B2 | DINOv2 mutual nearest neighbour: each patch's distance to the nearest patch in the *other* test images of the same category (a simplified MuSc) | Own code, ~60 lines | — | **Transductive** zero-shot: uses other unlabelled test images. Reported separately and never mixed with single-image methods | Phase 2 |

**Why these picks**
- **AnomalyCLIP over WinCLIP or Tipsomaly.** It's the most widely reproduced CLIP method, MIT-licensed,
  with weights in the repo, and it publishes HeadCT and BrainMRI numbers. WinCLIP has no released
  official code (only anomalib's re-implementation). Tipsomaly is new, has 13 stars, depends on
  external TIPS weights, and publishes its numbers only as an image.
- **AnomalyVFM over MuSc.** MuSc is MIT and training-free but transductive: it scores each image
  against the other test images. That's essentially baseline B2, so the two rows would overlap.
  AnomalyVFM scores one image at a time, which the demo needs, and has a DINOv2 checkpoint. MuSc is
  the fallback.
- **AnomalyVFM DINOv2 as the main foundation-model row, RADIO as an extra.** RADIO is distilled from
  CLIP, DINOv2 and SAM teachers, so it partly contains CLIP and blurs the family boundary. DINOv2 is
  purely self-supervised. RADIO scores higher on MVTec AD, so I'd also run it as an extra row:
  inference only, cheap.
- **VisualAD and AnomalyCLIP share the same frozen backbone** (CLIP ViT-L/14@336). That's useful:
  comparing them isolates *text prompts vs learned visual tokens* with the backbone held fixed.

**Licence handling.** Repos without a licence (AnomalyVFM, VisualAD) are cloned at setup time into
the gitignored `third_party/` folder and called through thin wrappers. None of their code is copied
into this repo. All borrowed work is credited in the README.

### Published numbers to reproduce (image AUROC / pixel AUROC / PRO, %)

| Method | MVTec AD | VisA | MPDD | BMAD Brain | HeadCT | Source |
|---|---|---|---|---|---|---|
| AnomalyCLIP | 91.5 / 91.1 / 81.4 | 82.1 / 95.5 / 87.0 | — | 69.0 / 95.1 / 71.5 † | 93.4 (image only) | AnomalyCLIP paper Tables 1–2; † from VisualAD Table 1 ("Brain_AD", cited to BraTS; *assumed* to be the BMAD brain split, to confirm in Phase 1 by comparing image counts) |
| VisualAD (CLIP) | 92.2 / 90.8 / 87.5 | 84.7 / 95.8 / 91.0 ‡ | — | 80.8 / 95.2 / 79.5 | — | VisualAD repo README and paper Table 1 |
| AnomalyVFM DINOv2 | 87.4 / 91.6 / 84.5 | 89.3 / 94.5 / 84.5 | 83.4 / 96.2 / 87.4 | — | 88.2 (image only) | `results/dinov2/*/result.csv` in the AnomalyVFM repo |
| AnomalyVFM RADIO | 96.4 / 92.0 / 87.6 | 92.9 / 95.8 / 90.5 | 85.7 / 96.8 / 90.2 | — | 94.0 (image only) | `results/radio/*/result.csv` |

‡ Needs the MVTec-trained VisualAD checkpoint, which isn't released (see open item 2).
"—" means no published number to compare against. We still run the method there.
Papers call the localisation metric "PRO"; we compute AUPRO at FPR ≤ 0.3 and will check that it
matches their definition in Phase 2.

## 4. Datasets

| Dataset | Role | Domain | Labels | Licence | Source for Kaggle runs |
|---|---|---|---|---|---|
| MVTec AD | benchmark; fusion train/test fold | industrial | image + pixel | CC BY-NC-SA 4.0 | `ipythonx/mvtec-ad` (Kaggle mirror, 5.3 GB). Will check file counts against the official release |
| VisA | benchmark; fusion train/test fold | industrial | image + pixel | CC BY 4.0 | Official S3 tar (1.93 GB, link checked) or Kaggle mirror |
| MPDD | **test-only stress set**: metal parts under varied lighting, pose and background | industrial (harder) | image + pixel | CC BY-NC-SA 4.0 | Official SharePoint link in [stepanje/MPDD](https://github.com/stepanje/MPDD); Kaggle copy `alirezasalehy/mpdd-ad` |
| BMAD Brain MRI (BraTS 2021 slices) | medical; fusion train/test fold | medical | image + pixel | BMAD: CC BY-NC-SA; derived from BraTS 2021 | Official BMAD Google Drive, re-uploaded as a private Kaggle dataset |
| Head CT (hemorrhage) | medical test set | medical | **image only** (200 images) | not yet checked (Phase 1) | `felipekitamura/head-ct-hemorrhage` (original upload) |

Medical datasets without masks (Head CT) only get image-level metrics. That is stated in every table.

## 5. Protocol and integrity rules (summary)

- No method or fusion head ever sees an image from the dataset it is scored on.
- Each method uses its paper's cross-dataset protocol (table above), recorded per output folder.
- **Fusion, leave-one-dataset-out.** Folds are MVTec AD, VisA and BMAD Brain (the datasets with
  pixel masks). **MPDD and Head CT are never used for fusion training.** They are scored by heads
  trained on the other three, in the role the spec gave the custom dataset.
- Hyperparameters are chosen on a validation split of the training folds only.
- A method counts as reproduced if its image AUROC is within 2 points of the paper on the same
  dataset and protocol. Larger gaps are logged with a likely cause.
- Every reported average comes with a 95% bootstrap CI over images (1,000 resamples).

## 6. Compute plan

- **Local, RTX 3050 6 GB:** development, unit tests, small runs. Inference fits in memory (≈2 GB per image for AnomalyVFM-DINOv2).
- **Kaggle T4 16 GB:** full benchmark runs, with data stored as Kaggle datasets. Outputs are cached per image so a dropped session can resume.
- **Known issue:** AnomalyVFM runs in **bfloat16**. The T4 (Turing) doesn't support bf16 natively, so the Phase 2 wrapper will run it in fp16 or fp32 there and check the scores match the bf16 run.

## 7. Success criteria

1. Phase 2: each single-image method's MVTec AD (and where possible VisA) image AUROC is within 2 points of its paper, or the gap is explained.
2. Phase 4: a complete method × dataset × metric table with 95% CIs, regenerated from cache with one command.
3. Phase 5: 3–5 findings in `analysis/FINDINGS.md`, each backed by a plot or table and a CI.
4. Phase 6: an ablation table where every number comes from a held-out dataset. A negative result is acceptable, reported against the oracle upper bound.
5. Phase 7: a public Hugging Face Space that processes a new photo in under 30 s.
6. Phase 8: a fresh clone reproduces the main table from cached outputs using the documented commands.

## 8. Open items needing a decision

1. **AnomalyVFM backbone.** Main row DINOv2 plus RADIO as an extra row (recommended), or DINOv2 only?
2. **VisualAD on VisA.** The MVTec-trained checkpoint isn't released. Options: (a) train it on Kaggle with the official script, at reduced batch size if 16 GB isn't enough (the paper used 24 GB), or (b) report VisualAD on VisA as N/A. Recommended: try (a), fall back to (b).
3. **No-licence repos.** AnomalyVFM and VisualAD can be used for research via wrappers, but not redistributed. The HF demo would download their weights at runtime. Is that acceptable to you?
