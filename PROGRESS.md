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
  - AnomalyVFM-RADIO, AnomalyCLIP, VisualAD: not yet run (the auto-mode permission check blocked running third-party code; waiting for Addy's approval).
- Wrote `docs/brief.md`.

### Decisions
- CLIP family = AnomalyCLIP; foundation-model family = AnomalyVFM (DINOv2, plus RADIO as an extra row, pending approval); language-free = VisualAD.
  Baselines: B1 plain CLIP prompt score (single image), B2 DINOv2 mutual-NN (transductive, reported separately).
- Third-party repos are cloned into gitignored `third_party/` and never vendored (two of them have no licence).
- Environment: Python 3.12 venv at `D:\Projects\envs\zsad`; `torch==2.8.0+cu128`, `torchvision==0.23.0`.
  AnomalyVFM needs `huggingface_hub==0.36.2` and `transformers==4.56.1`. With huggingface_hub 2.x its
  `from_pretrained` drops the config and fails with a shape mismatch.

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
