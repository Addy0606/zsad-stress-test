# Resume bullets and 2-minute demo script

Every number below is measured in this repo; the source of each is given in brackets so it can be checked.

## Resume

**Zero-Shot Anomaly Detection Stress Test** · Python, PyTorch, scikit-learn, Gradio · github.com/Addy0606/zsad-stress-test

- Benchmarked 4 zero-shot anomaly detectors from 3 model families (CLIP-based, language-free ViT, vision foundation
  model) plus 2 baselines on 5 industrial and medical datasets (8,260 test images) under one harness, with 95%
  bootstrap CIs on every result and a one-command rebuild of all tables from published cached outputs.
- Found that the strongest detector's anomaly map points away from the defect in 36–54% of the industrial images it
  correctly flags, and that tiny defects and missing parts are blind spots that differ by family.
- Showed with leave-one-dataset-out evaluation that averaging normalised anomaly maps improves localisation
  (AUPRO +1.1 to +2.3 points on all 4 held-out datasets), while learned image-level fusion does not beat the best
  single model; shipped a Hugging Face demo that runs all 4 detectors on a new photo in 10–20 s.

Sources: 8,260 = test images in results/benchmark.md (cost table); 36–54% = analysis/FINDINGS.md finding 2
(RADIO, 36.0–53.7%); +1.1 to +2.3 = results/fusion.md (AUPRO, mean of normalised maps); 10–20 s =
PROGRESS.md Phase 7 (9.5–18.5 s measured through the public API).

## 2-minute demo video script

| Time | On screen | Say |
|---|---|---|
| 0:00–0:15 | README title, demo GIF | "Zero-shot anomaly detectors are supposed to find defects on objects they've never seen. Papers report them one at a time, as averages. I put one detector from each of three families through the same harness to see where they actually fail." |
| 0:15–0:35 | Pipeline figure | "Five datasets: MVTec AD, VisA, a harder metal-parts set called MPDD that I only ever test on, and two medical sets. Four detectors, two baselines, 8,260 test images. Every number has a 95% bootstrap confidence interval, and no method is ever trained on the dataset it's scored on." |
| 0:35–0:55 | Main results table | "AnomalyVFM with a RADIO backbone has the best image-level score on every dataset, even though it was trained only on synthetic defects. But localisation is a different ranking: VisualAD's maps are best." |
| 0:55–1:15 | Figure "right image, wrong place" | "Here's the surprising part. Among anomalies RADIO correctly flags, its map's hottest point is off the defect in 36 to 54% of industrial images. Right image, wrong place. And the reverse happens too: AnomalyCLIP's image score is near chance on brain MRI, but its own map scores 85." |
| 1:15–1:35 | Fusion figure | "The methods miss different images, so I tried fusing them, trained leave-one-dataset-out. Fusing image scores doesn't beat the best single model on industrial data. That's a negative result. But simply averaging the normalised maps improves localisation on all four held-out datasets." |
| 1:35–1:55 | Live demo on the Space, upload a photo | "Here's the demo on Hugging Face. I upload a photo; all four detectors run on a free GPU in about 10 to 20 seconds. Each map, the fused map, and a verdict against a threshold set at 5% false alarms on normal parts." |
| 1:55–2:00 | GitHub repo | "Everything rebuilds from cached outputs with one command. Code and results are on GitHub." |
