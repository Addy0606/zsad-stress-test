"""CLIP-based family: AnomalyCLIP (Zhou et al., ICLR 2024), wrapping the official code (MIT).

Learned object-agnostic text prompts ("normal [object]" / "damaged [object]") are matched against
CLIP ViT-L/14@336 patch features. Prompts were trained on one industrial dataset's test set:
the MVTec-trained checkpoint is used for every target except MVTec AD itself, which uses the
VisA-trained one (the paper's cross-dataset protocol).

Matches the official test.py: 518 px input, layers 6/12/18/24 (configurable), DPAM from layer 20,
image score = softmax probability of the "anomalous" prompt for the global image embedding,
map = sum over layers of patch-prompt similarity, Gaussian-smoothed with sigma 4.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from methods.base import ZeroShotDetector, resize_map, to_pil, use_third_party

PROMPT_PARAMS = {
    "Prompt_length": 12,
    "learnabel_text_embedding_depth": 9,
    "learnabel_text_embedding_length": 4,
}


class AnomalyCLIP(ZeroShotDetector):
    name = "anomalyclip"
    family = "clip"

    def __init__(
        self,
        checkpoint: str,
        image_size: int = 518,
        sigma: float = 4.0,
        features_list: tuple[int, ...] = (6, 12, 18, 24),
        clip_cache: str | None = None,
        device: str = "cuda",
    ) -> None:
        """features_list: ViT layers whose patch features are scored. The paper and test.py default use
        6/12/18/24; the repo's test.sh passes only 24. Kept configurable; the configs use [24], as in the
        authors' official test script."""
        super().__init__(device)
        self.checkpoint, self.image_size, self.sigma, self.clip_cache = (
            checkpoint,
            image_size,
            sigma,
            clip_cache,
        )
        self.features_list = list(features_list)

    def setup(self, device: str) -> None:
        self.device = device
        use_third_party("AnomalyCLIP")
        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

        self._lib = AnomalyCLIP_lib
        model, _ = AnomalyCLIP_lib.load(
            "ViT-L/14@336px", device=device, design_details=PROMPT_PARAMS, download_root=self.clip_cache
        )
        model.eval()
        self.preprocess, _ = get_transform(argparse.Namespace(image_size=self.image_size))

        learner = AnomalyCLIP_PromptLearner(model.to("cpu"), PROMPT_PARAMS)
        learner.load_state_dict(torch.load(self.checkpoint, map_location="cpu")["prompt_learner"])
        learner.to(device)
        model.to(device)
        model.visual.DAPM_replace(DPAM_layer=20)
        with torch.no_grad():  # text features don't depend on the image: compute once
            prompts, tokenized, compound = learner(cls_id=None)
            text = model.encode_text_learn(prompts, tokenized, compound).float()
            text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
            self.text_features = text / text.norm(dim=-1, keepdim=True)
        self.model = model

    @torch.no_grad()
    def predict(self, image: torch.Tensor, category: str) -> tuple[float, np.ndarray]:
        x = self.preprocess(to_pil(image)).unsqueeze(0).to(self.device)
        img_feat, patch_feats = self.model.encode_image(x, self.features_list, DPAM_layer=20)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        probs = ((img_feat @ self.text_features.permute(0, 2, 1)) / 0.07).softmax(-1)
        score = float(probs[0, 0, 1])

        maps = []
        for pf in patch_feats:
            pf = pf / pf.norm(dim=-1, keepdim=True)
            sim, _ = self._lib.compute_similarity(pf, self.text_features[0])
            sim_map = self._lib.get_similarity_map(sim[:, 1:, :], self.image_size)
            maps.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0)
        amap = torch.stack(maps).sum(0)[0].cpu().numpy()
        amap = gaussian_filter(amap, sigma=self.sigma)
        return score, resize_map(amap, tuple(image.shape[1:]))
