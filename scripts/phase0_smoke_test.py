"""Phase 0 smoke test: does each candidate method load its released weights and run
one forward pass on a single image? Records peak GPU memory and time per image.

This is NOT an accuracy check (that is Phase 2). It only answers "repo exists,
weights download, runs on my GPU".

Usage (one method per process so peak-memory numbers don't mix):
    python scripts/phase0_smoke_test.py --method anomalyclip --image path/to/img.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
CLIP_CACHE = os.environ.get("CLIP_CACHE", str(Path.home() / ".cache" / "clip"))


def _use_repo(name: str) -> Path:
    """Put a third-party repo first on sys.path and cd into it (their code uses relative imports/paths)."""
    path = THIRD_PARTY / name
    sys.path.insert(0, str(path))
    os.chdir(path)
    return path


def run_anomalyclip(image: Image.Image, checkpoint: str) -> tuple[float, np.ndarray]:
    _use_repo("AnomalyCLIP")
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    # Hyperparameters from the official test.sh
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    args = argparse.Namespace(image_size=518)
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cuda", design_details=params, download_root=CLIP_CACHE)
    model.eval()
    preprocess, _ = get_transform(args)

    learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    learner.load_state_dict(torch.load(checkpoint, map_location="cpu")["prompt_learner"])
    learner.to("cuda")
    model.to("cuda")
    model.visual.DAPM_replace(DPAM_layer=20)

    with torch.no_grad():
        prompts, tokenized, compound = learner(cls_id=None)
        text = model.encode_text_learn(prompts, tokenized, compound).float()
        text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
        text = text / text.norm(dim=-1, keepdim=True)

        x = preprocess(image).unsqueeze(0).cuda()
        img_feat, patch_feats = model.encode_image(x, [6, 12, 18, 24], DPAM_layer=20)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        probs = ((img_feat @ text.permute(0, 2, 1)) / 0.07).softmax(-1)
        score = probs[0, 0, 1].item()

        maps = []
        for pf in patch_feats:
            pf = pf / pf.norm(dim=-1, keepdim=True)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pf, text[0])
            sim_map = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], 518)
            maps.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0)
        amap = torch.stack(maps).sum(0)[0].cpu().numpy()
    return score, amap


def run_visualad(image: Image.Image, checkpoint: str) -> tuple[float, np.ndarray]:
    _use_repo("VisualAD")
    import torch.nn.functional as F
    import VisualAD_lib
    from utils.anomaly_detection import generate_anomaly_map_from_tokens
    from utils.feature_transform import create_feature_transform
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
    from utils.transforms import get_transform

    ckpt = torch.load(checkpoint, map_location="cuda")
    size = ckpt.get("image_size", 518)
    layers = ckpt.get("features_list", [6, 12, 18, 24])
    preprocess, _ = get_transform(argparse.Namespace(image_size=size))

    model, _ = VisualAD_lib.load(ckpt.get("backbone", "ViT-L/14@336px"), device="cuda", download_root=CLIP_CACHE)
    model.eval().to("cuda")
    dim = model.visual.embed_dim
    model.visual.anomaly_token.data = ckpt["anomaly_token"].cuda()
    model.visual.normal_token.data = ckpt["normal_token"].cuda()
    model.visual.ln_post.load_state_dict({"weight": ckpt["ln_post_weight"], "bias": ckpt["ln_post_bias"]})

    transforms = {}
    for name, sd in ckpt.get("layer_transforms", {}).items():
        t = create_feature_transform("mlp", dim, sd["mlp.0.weight"].shape[0], dim, 0.0).cuda()
        t.load_state_dict(sd)
        transforms[name] = t.eval()
    cfg = ckpt.get("cross_attn_config", {})
    cross = build_layer_adaptive_cross_attention(
        layers=layers,
        embed_dim=dim,
        num_anchors=cfg.get("num_anchors", 4),
        dropout=cfg.get("dropout", 0.1),
        res_scale_init=cfg.get("res_scale_init", 0.01),
    ).cuda()
    cross.load_state_dict(ckpt["cross_attn"])
    cross.eval()

    with torch.no_grad():
        x = preprocess(image).unsqueeze(0).cuda()
        out = model.encode_image(x, layers)
        start = out["patch_start_idx"]
        patches = [pt[:, start:, :] for pt in out["patch_tokens"]]
        adapted = cross(out["anomaly_features"], out["normal_features"], patches, layers)
        maps = []
        for i, pf in enumerate(out["patch_tokens"]):
            key = f"layer_{layers[i]}"
            if key in transforms:
                b, n, d = pf.shape
                pf = transforms[key](pf.view(-1, d)).view(b, n, d)
            maps.append(
                generate_anomaly_map_from_tokens(
                    F.normalize(adapted[i]["anomaly"], dim=1, eps=1e-8),
                    F.normalize(adapted[i]["normal"], dim=1, eps=1e-8),
                    pf[:, start:, :],
                    size,
                )
            )
        amap = torch.stack(maps).sum(0)[0].cpu().numpy()
    k = max(1, round(0.01 * amap.size))  # paper: image score = mean of top 1% pixels (industrial)
    score = float(np.sort(amap.ravel())[-k:].mean())
    return score, amap


def run_anomalyvfm(image: Image.Image, variant: str) -> tuple[float, np.ndarray]:
    _use_repo("AnomalyVFM")
    from hf_model import AnomalyVFM

    model = AnomalyVFM.from_pretrained(f"MaticFuc/anomalyvfm_{variant}").cuda().eval()
    tf = model.model.get_img_transform()  # backbone-specific resize + normalisation, as in the README
    score, mask = model(tf(image).unsqueeze(0).cuda())
    return float(score.flatten()[0]), mask[0, 0].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        required=True,
        choices=["anomalyclip", "visualad", "anomalyvfm_radio", "anomalyvfm_dinov2"],
    )
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "phase0_smoke"))
    a = p.parse_args()

    image = Image.open(a.image).convert("RGB")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    if a.method == "anomalyclip":
        score, amap = run_anomalyclip(image, a.checkpoint)
    elif a.method == "visualad":
        score, amap = run_visualad(image, a.checkpoint)
    else:
        score, amap = run_anomalyvfm(image, a.method.split("_", 1)[1])
    torch.cuda.synchronize()
    total = time.time() - t0  # includes model load; per-image timing is measured properly in Phase 4

    result = {
        "method": a.method,
        "image": a.image,
        "image_score": score,
        "map_shape": list(amap.shape),
        "map_min": float(amap.min()),
        "map_max": float(amap.max()),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "load_plus_infer_s": round(total, 1),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    print(json.dumps(result, indent=2))
    (out_dir / f"{a.method}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / f"{a.method}_map.npz", amap=amap)


if __name__ == "__main__":
    main()
