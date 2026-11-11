"""Gradio demo (Phase 7): upload a photo, see where each zero-shot detector thinks the defect is.

For one photo it runs the four detectors benchmarked in this project, shows each one's anomaly map over the
photo, and the fused map (the Phase 6 "mean of normalised maps" head). Optionally it also runs baseline B1,
the only method that reads the category name.

How the pieces fit:
- Detectors come from methods/ with their benchmark configs (configs/methods/). The checkpoint is the one the
  benchmark used for datasets the detector was NOT trained on ("default" in checkpoint_by_target).
- Every map is resized to 256x256 (the benchmark's evaluation size) and put through the normaliser fitted on
  the training datasets in Phase 6 (app/fusion_normalizer.npz, made by fusion/export_normalizer.py): a value
  becomes "the fraction of training pixels with a lower value", in [0, 1]. The fused map is the mean of the four.
- The heatmaps are coloured on that common [0, 1] scale, so colours mean the same thing for every method.
- Image scores are reported against normal reference images (app/score_reference.json, built by
  app/build_reference.py): "higher than X% of normal images".

Third-party code and weights are downloaded when the app starts (two of the repos have no licence, so they are
never copied into this repo or the Space).

Run locally:  python app/app.py        On the Space: app.py at the root (scripts/deploy_space.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:  # Hugging Face ZeroGPU: a GPU is attached only while a @spaces.GPU function runs
    import spaces

    on_gpu = spaces.GPU(duration=120)
    ZERO_GPU = True
except ImportError:  # anywhere else: run as is
    ZERO_GPU = False

    def on_gpu(fn):
        return fn


import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "methods").is_dir() else HERE.parent  # Space: everything at the root; repo: app/ inside
DATA = ROOT / "app"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CLIP_CACHE", str(Path.home() / ".cache" / "clip"))  # AnomalyCLIP's own default is unusable

from fusion.normalize import QuantileNormalizer  # noqa: E402
from methods.base import resize_map, topk_mean  # noqa: E402
from methods.registry import build_method, resolve_method_config  # noqa: E402

DETECTORS = {  # same order as the fusion normaliser's columns
    "anomalyclip": "AnomalyCLIP (CLIP + learned text prompts)",
    "visualad": "VisualAD (frozen CLIP ViT + learned visual tokens, no text)",
    "anomalyvfm_dinov2": "AnomalyVFM, DINOv2 backbone",
    "anomalyvfm_radio": "AnomalyVFM, RADIO backbone",
}
B1 = "clip_prompt"
EVAL_SIZE = 256
MAX_SIDE = 1024  # phone photos are shrunk first; every method resizes internally anyway
DEVICE = "cuda" if ZERO_GPU or torch.cuda.is_available() else "cpu"  # ZeroGPU: models go to "cuda" at startup
VISUALAD_TOPK = {"industrial": 0.01, "medical": 0.05}  # configs/methods/visualad.yaml, params_by_domain

# blue -> cyan -> yellow -> red, for values in [0, 1]
COLORS = np.array([[0, 0, 130], [0, 120, 255], [0, 230, 230], [255, 230, 0], [230, 0, 0]], dtype=np.float32)


def setup_third_party() -> None:
    if not (ROOT / "third_party" / "AnomalyVFM").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "setup_third_party.py")], check=True)


def load_normalizer() -> QuantileNormalizer:
    z = np.load(DATA / "fusion_normalizer.npz")
    assert tuple(z["detectors"]) == tuple(DETECTORS), "normaliser columns don't match the detector order"
    norm = QuantileNormalizer(n_quantiles=len(z["grid"]))
    norm.grid_ = z["grid"]
    return norm


def load_models() -> dict:
    models = {}
    for name in (*DETECTORS, B1):
        t0 = time.perf_counter()
        det = build_method(resolve_method_config(name, target="new_photo", domain="industrial"))
        det.setup(DEVICE)
        models[name] = det
        print(f"loaded {name} in {time.perf_counter() - t0:.0f} s", flush=True)
    return models


setup_third_party()
NORM = load_normalizer()
REFERENCE = json.loads((DATA / "score_reference.json").read_text(encoding="utf-8"))
MODELS = load_models()


def to_tensor(photo: Image.Image) -> torch.Tensor:
    photo = photo.convert("RGB")
    if max(photo.size) > MAX_SIDE:
        photo.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
    return TF.to_tensor(photo)


def overlay(photo: np.ndarray, values: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Heatmap of `values` (in [0, 1], any size) blended over `photo` (H, W, 3 uint8)."""
    v = resize_map(values, photo.shape[:2]).clip(0, 1) * (len(COLORS) - 1)
    lo = np.floor(v).astype(int).clip(0, len(COLORS) - 2)
    frac = (v - lo)[..., None]
    heat = COLORS[lo] * (1 - frac) + COLORS[lo + 1] * frac
    return (photo * (1 - alpha) + heat * alpha).astype(np.uint8)


def percentile(score: float, domain: str, key: str) -> float:
    """Percentage of normal reference images with a lower score."""
    ref = np.asarray(REFERENCE[domain][key])
    return 100.0 * np.searchsorted(ref, score, side="left") / len(ref)


def threshold(domain: str, key: str, fpr: float = 0.05) -> float:
    """Score above which an image is flagged: at most `fpr` of normal reference images lie above it.
    Same rule as analysis/image_level.py (fpr_threshold) in the failure analysis."""
    return float(np.quantile(REFERENCE[domain][key], 1 - fpr, method="inverted_cdf"))


@on_gpu
@torch.inference_mode()
def analyse(photo: Image.Image | None, category: str, domain: str, run_b1: bool):
    if photo is None:
        raise gr.Error("Upload a photo first.")
    category = category.strip() or "object"
    x = to_tensor(photo)
    shown = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    MODELS["visualad"].topk_ratio = VISUALAD_TOPK[domain]  # its image score uses a larger top-k on medical images

    rows, gallery, maps = [], [], []
    t_all = time.perf_counter()
    for name, label in DETECTORS.items():
        t0 = time.perf_counter()
        score, amap = MODELS[name].predict(x, category)
        seconds = time.perf_counter() - t0
        maps.append(resize_map(amap, (EVAL_SIZE, EVAL_SIZE)))
        rows.append((label, name, score, seconds))
    normalised = NORM.transform(np.stack(maps).reshape(len(DETECTORS), -1).T)  # (256*256, 4), each in [0, 1]
    for i, (label, *_rest) in enumerate(rows):
        gallery.append((overlay(shown, normalised[:, i].reshape(EVAL_SIZE, EVAL_SIZE)), label))
    fused = normalised.mean(1).reshape(EVAL_SIZE, EVAL_SIZE)
    fused_score = topk_mean(fused, 0.01)
    gallery.insert(0, (overlay(shown, fused), "Fused map: mean of the four normalised maps"))
    rows.append(
        (
            "Fused map, mean of its top 1% (for locating defects; a weak image score in Phase 6)",
            "fused",
            fused_score,
            0.0,
        )
    )

    if run_b1:
        t0 = time.perf_counter()
        score, amap = MODELS[B1].predict(x, category)
        seconds = time.perf_counter() - t0
        a = resize_map(amap, (EVAL_SIZE, EVAL_SIZE))
        a = (a - a.min()) / (a.max() - a.min() + 1e-12)  # not in the fusion normaliser: own range
        gallery.append((overlay(shown, a), f"B1 CLIP prompt '{category}' (colours: this map's own range)"))
        rows.append((f"B1 CLIP prompt, category '{category}'", B1, score, seconds))
    total = time.perf_counter() - t_all

    ref = REFERENCE[domain]
    table = [
        "| Method | Raw score | Flag threshold | Verdict | Higher than this % of normal reference images | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for label, key, score, seconds in rows:
        t = threshold(domain, key)
        verdict = "**defect flagged**" if score > t else "looks normal"
        table.append(
            f"| {label} | {score:.4f} | {t:.4f} | {verdict} | {percentile(score, domain, key):.0f}% | {seconds:.1f} |"
        )
    table += [
        "",
        f"Model time: **{total:.1f} s** on {DEVICE.upper()} (the page also waits for a shared GPU and to send the "
        "images, usually several seconds).",
        "",
        "**Flag threshold:** each method's raw score is on its own scale, so each has its own threshold: the score "
        "that only 5% of normal reference images exceed. A photo scoring above it is flagged, accepting about 5% "
        "false alarms on normal parts. Reference: normal test images of "
        f"{' + '.join(ref['datasets'])} (fused map: {' + '.join(ref['fused_datasets'])}), scored in this project's "
        "benchmark with the same checkpoints. Thresholds depend on that reference: for objects very unlike it, "
        "treat the verdict as a hint and look at the maps.",
    ]
    return gallery, "\n".join(table)


INTRO = """
# Zero-shot anomaly detection: four detectors, one photo

Upload a photo of an object (or a medical scan). Four zero-shot detectors from three families each mark where
they think a defect is. None of them has seen your object before. The **fused map** averages their normalised
maps: in this project's held-out tests it localised defects better than any single detector
(AUPRO +1.1 to +2.3 points on four held-out datasets). For the image-level "is it defective?" call,
AnomalyVFM RADIO's own score was the best single signal, and fusing image scores did not beat it on the
industrial datasets.

Runs all four detectors on a shared free GPU (Hugging Face ZeroGPU); the time each one took is shown with the
scores. If the Space was asleep, the first request also waits for it to start and download the models.

Colours: blue = ordinary for that detector, red = higher than almost all of its values on pixels of the three
training datasets (MVTec AD, VisA, BMAD Brain MRI). The four detectors don't read the category name; it is only
used by the optional B1 baseline.
Code, benchmark and failure analysis: [github.com/Addy0606/zsad-stress-test](https://github.com/Addy0606/zsad-stress-test).
"""

CREDITS = """
**Credits.** [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) (MIT licence),
[VisualAD](https://github.com/7HHHHH/VisualAD) and [AnomalyVFM](https://github.com/MaticFuc/AnomalyVFM) (no licence
file: their code and weights are downloaded from the authors when the app starts and are not redistributed here).
Reference scores come from MPDD (Jezek et al.), BMAD Brain MRI (Bao et al., BraTS 2021) and Head CT (Kitamura).
"""


def processing_note() -> str:
    """How the photo is resized, with each method's input size read from the loaded models."""
    names = {**DETECTORS, B1: "B1 CLIP prompt"}
    sizes = "; ".join(f"{names[n].split(' (')[0]}: {m.image_size}x{m.image_size}" for n, m in MODELS.items())
    return f"""
**How your photo is processed: nothing is cropped.**
1. If its longest side is over {MAX_SIDE} px, the photo is shrunk to {MAX_SIDE} px, keeping its shape.
2. Each method stretches the *whole* photo to its own square input ({sizes}). A tall or wide photo is
   squashed, not cut, so every part of it is analysed.
3. Each map is stretched back to the photo's shape for display. For the fused map and the colours, the maps are
   compared at {EVAL_SIZE}x{EVAL_SIZE}, as in the benchmark.

Tip: a roughly square photo with the object filling most of it squashes least, and a plain background avoids false
alarms (the detectors only know "unusual compared with their training images", and those have clean backgrounds).
"""


with gr.Blocks(title="Zero-shot anomaly detection stress test") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(type="pil", label="Photo")
            domain = gr.Radio(["industrial", "medical"], value="industrial", label="Domain")
            category = gr.Textbox(value="object", label="Category name (used only by B1)")
            run_b1 = gr.Checkbox(value=False, label="Also run baseline B1 (plain CLIP prompt, adds a few seconds)")
            button = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Anomaly maps", columns=3, height="auto", object_fit="contain")
            scores = gr.Markdown()
    button.click(analyse, [photo, category, domain, run_b1], [gallery, scores])
    with gr.Accordion("How your photo is processed (nothing is cropped)", open=False):
        gr.Markdown(processing_note())
    gr.Markdown(CREDITS)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
