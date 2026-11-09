"""Vision-foundation-model family: AnomalyVFM (Fučka et al., CVPR 2026), wrapping the official code.

A frozen foundation backbone (DINOv2 ViT-L/14 or RADIO v2.5-L) with low-rank (DoRA) adapters,
a small decoder for the anomaly map and a linear head for the image score. Trained only on
synthetic images, never on real benchmark images, so it is zero-shot on every dataset here.

Matches the official test.py: backbone-specific transform (672 px DINOv2, 768 px RADIO),
fp16 autocast, image score = sigmoid(predictor(summary token)),
map = sigmoid(decoder(patch features)), resized, then 5x5 mean filter.

Note: the repo's Hugging Face `forward` uses bfloat16, but its evaluation script uses float16;
we follow the evaluation script (float16 also runs natively on a T4, bfloat16 does not).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import ZeroShotDetector, resize_map, to_pil, use_third_party


class AnomalyVFM(ZeroShotDetector):
    family = "vfm"

    def __init__(self, backbone: str = "dinov2", mean_kernel: int = 5, device: str = "cuda") -> None:
        super().__init__(device)
        if backbone not in ("dinov2", "radio"):
            raise ValueError("backbone must be 'dinov2' or 'radio'")
        self.backbone, self.mean_kernel = backbone, mean_kernel
        self.name = f"anomalyvfm_{backbone}"

    def setup(self, device: str) -> None:
        self.device = device
        use_third_party("AnomalyVFM")
        from hf_model import AnomalyVFM as HFAnomalyVFM

        self.model = HFAnomalyVFM.from_pretrained(f"MaticFuc/anomalyvfm_{self.backbone}").to(device).eval()
        self.image_size = self.model.config.image_size
        self.transform = self.model.model.get_img_transform()
        self.smooth = nn.AvgPool2d(self.mean_kernel, 1, self.mean_kernel // 2)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, category: str) -> tuple[float, np.ndarray]:
        x = self.transform(to_pil(image)).unsqueeze(0).to(self.device)
        m = self.model
        with torch.autocast("cuda", dtype=torch.float16):
            summary, feats = m.model(x)
            feats = feats.permute(0, 2, 1).reshape(1, -1, m.feat_size, m.feat_size)
            mask, _ = m.decoder(feats)
            score = m.predictor(summary).squeeze().sigmoid()
        mask = mask.float().sigmoid()
        # Official eval: resize to the mask size, then mean filter. We resize to the model input
        # size (the eval-size resize happens later, the same way for every method).
        mask = F.interpolate(mask, (self.image_size, self.image_size))
        amap = self.smooth(mask[:, 0])[0].cpu().numpy()
        return float(score), resize_map(amap, tuple(image.shape[1:]))
