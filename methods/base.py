"""The ZeroShotDetector interface every method implements, plus small shared helpers.

    det = SomeDetector(**config)
    det.setup("cuda")
    score, amap = det.predict(image, category)   # image: 3 x H x W float tensor in [0, 1]
                                                 # amap: H x W numpy array (same H, W as the input)

Transductive methods (which score an image against the *other unlabelled test images* of the
same category, like baseline B2) set `transductive = True`; the benchmark runner then calls
`prepare_category(images)` with all test images of a category before predicting them.
"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"

# Module names that several third-party repos define at top level (utils.py, models/, ...).
# They are dropped from the import cache before importing another repo, so the right one loads.
_CLASHING_MODULES = (
    "utils",
    "dataset",
    "datasets",
    "models",
    "logger",
    "decoder",
    "loss",
    "metrics",
    "prompt_ensemble",
    "hf_model",
    "peft_local",
    "radio_local",
)


class ZeroShotDetector(ABC):
    name: str = "base"
    family: str = "baseline"  # "clip", "vfm", "langfree" or "baseline"
    image_size: int = 518  # resolution the model runs at; fixed per method in configs/methods/
    transductive: bool = False

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    @abstractmethod
    def setup(self, device: str) -> None:
        """Load weights onto `device`."""

    @abstractmethod
    def predict(self, image: torch.Tensor, category: str) -> tuple[float, np.ndarray]:
        """Return (image score, anomaly map at the input image's resolution). Higher = more anomalous."""

    def prepare_category(self, images: list[torch.Tensor], category: str) -> None:  # noqa: B027 (optional hook)
        """Only for transductive methods: see all unlabelled test images of a category first."""

    def features(self, image: torch.Tensor) -> dict:
        """Optional pooled features for the fusion head (Phase 6)."""
        raise NotImplementedError(f"{self.name} does not provide features")


# ---------- helpers shared by the wrappers ----------


def use_third_party(repo: str) -> Path:
    """Put third_party/<repo> first on sys.path and forget clashing modules from other repos."""
    path = THIRD_PARTY / repo
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python scripts/setup_third_party.py`")
    for mod in list(sys.modules):
        if mod.split(".")[0] in _CLASHING_MODULES:
            del sys.modules[mod]
    sys.path = [p for p in sys.path if not p.startswith(str(THIRD_PARTY))]
    sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    return path


def to_pil(image: torch.Tensor) -> Image.Image:
    """3 x H x W float [0, 1] -> PIL RGB. Lossless for images that were loaded from 8-bit files,
    so each wrapper can reuse its repo's own PIL-based preprocessing exactly."""
    arr = (image.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def resize_map(amap: np.ndarray | torch.Tensor, size_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2-D map to (H, W)."""
    t = torch.as_tensor(np.asarray(amap, dtype=np.float32))[None, None]
    return F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def topk_mean(amap: np.ndarray, ratio: float) -> float:
    """Mean of the top `ratio` fraction of map values (at least one value)."""
    flat = np.sort(amap.ravel())
    k = max(1, int(round(ratio * flat.size)))
    return float(flat[-k:].mean())
