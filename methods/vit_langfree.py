"""Language-free family: VisualAD (Hou et al., CVPR 2026), wrapping the official code.

No text encoder: two learned tokens (anomaly, normal) are added to a frozen CLIP ViT-L/14@336
*vision* encoder, plus a spatial cross-attention module and per-layer feature transforms.
Each patch is scored by how much closer it is to the anomaly token than to the normal token.
Trained on VisA (released checkpoint) and tested on everything except VisA; for VisA a
checkpoint trained on MVTec AD is needed (not released; we train it ourselves).

Matches the official test.py: 518 px, layers 6/12/18/24, map = sum over layers,
Gaussian sigma 4; image score = mean of the top-k% map values (paper: 1% industrial,
5% medical, per the authors in GitHub issue #9; set in the method config).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from methods.base import ZeroShotDetector, resize_map, to_pil, topk_mean, use_third_party


class VisualAD(ZeroShotDetector):
    name = "visualad"
    family = "langfree"

    def __init__(
        self,
        checkpoint: str,
        topk_ratio: float = 0.01,
        sigma: float = 4.0,
        clip_cache: str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(device)
        self.checkpoint, self.topk_ratio, self.sigma, self.clip_cache = (
            checkpoint,
            topk_ratio,
            sigma,
            clip_cache,
        )

    def setup(self, device: str) -> None:
        self.device = device
        use_third_party("VisualAD")
        import VisualAD_lib
        from utils.anomaly_detection import generate_anomaly_map_from_tokens
        from utils.feature_transform import create_feature_transform
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
        from utils.transforms import get_transform

        self._map_fn = generate_anomaly_map_from_tokens
        ckpt = torch.load(self.checkpoint, map_location=device)
        self.image_size = ckpt.get("image_size", 518)
        self.layers = ckpt.get("features_list", [6, 12, 18, 24])
        self.preprocess, _ = get_transform(argparse.Namespace(image_size=self.image_size))

        model, _ = VisualAD_lib.load(
            ckpt.get("backbone", "ViT-L/14@336px"), device=device, download_root=self.clip_cache
        )
        model.eval().to(device)
        dim = model.visual.embed_dim
        model.visual.anomaly_token.data = ckpt["anomaly_token"].to(device)
        model.visual.normal_token.data = ckpt["normal_token"].to(device)
        model.visual.ln_post.load_state_dict({"weight": ckpt["ln_post_weight"], "bias": ckpt["ln_post_bias"]})

        self.transforms = {}
        for name, sd in ckpt.get("layer_transforms", {}).items():
            t = create_feature_transform("mlp", dim, sd["mlp.0.weight"].shape[0], dim, 0.0).to(device)
            t.load_state_dict(sd)
            self.transforms[name] = t.eval()
        cfg = ckpt.get("cross_attn_config", {})
        self.cross = build_layer_adaptive_cross_attention(
            layers=self.layers,
            embed_dim=dim,
            num_anchors=cfg.get("num_anchors", 4),
            dropout=cfg.get("dropout", 0.1),
            res_scale_init=cfg.get("res_scale_init", 0.01),
        ).to(device)
        self.cross.load_state_dict(ckpt["cross_attn"])
        self.cross.eval()
        self.model = model

    @torch.no_grad()
    def predict(self, image: torch.Tensor, category: str) -> tuple[float, np.ndarray]:
        x = self.preprocess(to_pil(image)).unsqueeze(0).to(self.device)
        out = self.model.encode_image(x, self.layers)
        start = out["patch_start_idx"]
        patches = [pt[:, start:, :] for pt in out["patch_tokens"]]
        adapted = self.cross(out["anomaly_features"], out["normal_features"], patches, self.layers)
        maps = []
        for i, pf in enumerate(out["patch_tokens"]):
            key = f"layer_{self.layers[i]}"
            if key in self.transforms:
                b, n, d = pf.shape
                pf = self.transforms[key](pf.view(-1, d)).view(b, n, d)
            maps.append(
                self._map_fn(
                    F.normalize(adapted[i]["anomaly"], dim=1, eps=1e-8),
                    F.normalize(adapted[i]["normal"], dim=1, eps=1e-8),
                    pf[:, start:, :],
                    self.image_size,
                )
            )
        amap = torch.stack(maps).sum(0)[0].cpu().numpy()
        amap = gaussian_filter(amap, sigma=self.sigma)
        return topk_mean(amap, self.topk_ratio), resize_map(amap, tuple(image.shape[1:]))
