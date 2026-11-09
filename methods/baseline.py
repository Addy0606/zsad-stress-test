"""Two simple baselines, written from scratch so every line can be explained.

B1  CLIPPrompt: plain CLIP, no training. Compare the image (and each patch) with two text prompts,
    "a photo of a flawless {object}" vs "a photo of a damaged {object}". Pure single-image zero-shot.
    Patch-level CLIP features are known to align poorly with text; that weakness is the point
    of a baseline.

B2  DINOMutualNN: DINOv2 patch features, no text, no training. A patch is anomalous if it has
    no close match in the *other* test images of the same category, because defects are rare
    and differ between images while normal structure repeats (a simplified MuSc).
    TRANSDUCTIVE: it uses the unlabelled test set, so it is reported separately.
"""

from __future__ import annotations

import os
import random
import urllib.request
from collections.abc import Sequence

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from methods.base import ZeroShotDetector, resize_map, to_pil, topk_mean

# Human-readable names for prompts; anything not listed uses the category with "_" -> " ".
PROMPT_NAMES = {
    "head_ct": "head CT scan",
    "brain": "brain MRI scan",
    "metal_nut": "metal nut",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pipe_fryum": "fryum",
}
TEMPLATES = [
    "a photo of a {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of the {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
]
# Official OpenAI download (same file AnomalyCLIP and VisualAD load); SHA256 prefix is in the URL.
CLIP_L336_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
)
STATES = {
    "normal": ["flawless {}", "perfect {}", "{} without defect"],
    "anomalous": ["damaged {}", "broken {}", "{} with a defect"],
}


class CLIPPrompt(ZeroShotDetector):
    name = "clip_prompt"
    family = "baseline"
    image_size = 336

    def __init__(self, clip_weights: str, device: str = "cuda") -> None:
        super().__init__(device)
        self.clip_weights = clip_weights
        self._text_cache: dict[str, torch.Tensor] = {}

    def setup(self, device: str) -> None:
        """Load OpenAI CLIP ViT-L/14@336 (the same weight file AnomalyCLIP uses) into open_clip."""
        self.device = device
        if not os.path.exists(self.clip_weights):
            os.makedirs(os.path.dirname(self.clip_weights), exist_ok=True)
            urllib.request.urlretrieve(CLIP_L336_URL, self.clip_weights)
        sd = torch.jit.load(self.clip_weights, map_location="cpu").state_dict()
        for k in ("input_resolution", "context_length", "vocab_size"):
            sd.pop(k, None)
        model = open_clip.create_model("ViT-L-14-336", pretrained=None, force_quick_gelu=True)
        model.load_state_dict(sd)
        model.visual.output_tokens = True
        self.model = model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
        self.preprocess = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    @torch.no_grad()
    def _text_features(self, category: str) -> torch.Tensor:
        """(2, D): averaged, normalised embeddings for [normal, anomalous] prompts."""
        if category not in self._text_cache:
            obj = PROMPT_NAMES.get(category, category.replace("_", " "))
            feats = []
            for state in ("normal", "anomalous"):
                texts = [t.format(s.format(obj)) for s in STATES[state] for t in TEMPLATES]
                emb = self.model.encode_text(self.tokenizer(texts).to(self.device))
                emb = F.normalize(emb, dim=-1).mean(0)
                feats.append(F.normalize(emb, dim=-1))
            self._text_cache[category] = torch.stack(feats)
        return self._text_cache[category]

    @torch.no_grad()
    def predict(self, image: torch.Tensor, category: str) -> tuple[float, np.ndarray]:
        text = self._text_features(category)
        x = self.preprocess(to_pil(image)).unsqueeze(0).to(self.device)
        pooled, tokens = self.model.visual(x)
        pooled = F.normalize(pooled, dim=-1)
        patches = F.normalize(tokens @ self.model.visual.proj, dim=-1)  # (1, 576, D)
        scale = 100.0  # CLIP's logit scale
        score = float((scale * pooled @ text.T).softmax(-1)[0, 1])
        patch_prob = (scale * patches @ text.T).softmax(-1)[0, :, 1]  # P(anomalous) per patch
        side = int(patch_prob.numel() ** 0.5)
        amap = patch_prob.reshape(side, side).cpu().numpy()
        return score, resize_map(amap, tuple(image.shape[1:]))


class DINOMutualNN(ZeroShotDetector):
    name = "dino_mutual_nn"
    family = "baseline"
    transductive = True

    def __init__(
        self,
        image_size: int = 448,
        n_refs: int = 200,
        low_fraction: float = 0.3,
        topk_ratio: float = 0.01,
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        """
        n_refs:        reference images per category (other test images), capped for memory.
        low_fraction:  a patch's score is the mean of its lowest `low_fraction` nearest-neighbour
                       distances over the references, so a few anomalous references don't matter.
        topk_ratio:    image score = mean of the top fraction of patch scores.
        """
        super().__init__(device)
        self.image_size, self.n_refs, self.low_fraction = image_size, n_refs, low_fraction
        self.topk_ratio, self.seed = topk_ratio, seed
        self.refs: torch.Tensor | None = None

    def setup(self, device: str) -> None:
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(device).eval()
        self.preprocess = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def _patch_features(self, image: torch.Tensor) -> torch.Tensor:
        """(P, D) L2-normalised patch features, float16."""
        x = self.preprocess(to_pil(image)).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = self.model.forward_features(x)["x_norm_patchtokens"][0]
        return F.normalize(feats.float(), dim=-1).half()

    def prepare_category(self, images: Sequence[torch.Tensor], category: str) -> None:
        """Pick up to n_refs reference images (fixed seed) and cache their patch features on the GPU."""
        idx = list(range(len(images)))
        random.Random(self.seed).shuffle(idx)
        self.refs = torch.stack([self._patch_features(images[i]) for i in sorted(idx[: self.n_refs])])

    @torch.no_grad()
    def predict(self, image: torch.Tensor, category: str) -> tuple[float, np.ndarray]:
        if self.refs is None:
            raise RuntimeError("call prepare_category() first (transductive method)")
        q = self._patch_features(image)  # (P, D)
        # cosine distance from each query patch to its nearest patch in each reference image
        sims = torch.einsum("pd,rqd->rpq", q, self.refs)  # (R, P, Q)
        dist = 1.0 - sims.max(dim=2).values.float()  # (R, P)
        # never compare an image with itself (it may be one of the references)
        dist = dist[dist.mean(dim=1) > 1e-3]
        k = max(1, int(round(self.low_fraction * dist.shape[0])))
        patch_score = dist.topk(k, dim=0, largest=False).values.mean(0)  # (P,)
        side = int(patch_score.numel() ** 0.5)
        amap = patch_score.reshape(side, side).cpu().numpy()
        return topk_mean(amap, self.topk_ratio), resize_map(amap, tuple(image.shape[1:]))
