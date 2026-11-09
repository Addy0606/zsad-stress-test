"""The AnomalyDataset interface shared by every dataset in this project.

Each dataset subclass only has to implement `_scan()`, which returns a list of
`Record`s (where the files are and what the labels are). Loading images and
masks is done here, the same way for every dataset, so all loaders return
items with exactly the same keys and types:

    image        float32 tensor, 3 x H x W, RGB, values in [0, 1] (no normalisation;
                 each method normalises in its own way later)
    label        int, 0 = normal, 1 = anomalous
    mask         uint8 tensor, H x W, values in {0, 1}; all zeros for normal images
    category     str, object or organ class, e.g. "screw", "brain"
    domain       str, "industrial" or "medical"
    defect_type  str or None (None for normal images)
    path         str, source image path, used as the unique ID
    dataset      str, dataset name, e.g. "mvtec" (extra key, used for leave-one-dataset-out)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ITEM_KEYS = ("image", "label", "mask", "category", "domain", "defect_type", "path", "dataset")


@dataclass(frozen=True)
class Record:
    """Where one test image lives on disk and what its labels are."""

    image_path: str
    label: int
    category: str
    defect_type: str | None = None
    mask_path: str | None = None  # None for normal images, or when the dataset has no masks


class AnomalyDataset(Dataset):
    """Base class: subclasses implement `_scan()`; everything else is shared.

    Args:
        root: dataset root folder (layout depends on the subclass).
        categories: optional list of categories to keep; None keeps all.
        image_size: optional square size to resize to (image: bicubic, mask: nearest).
            None keeps the native resolution. The benchmark fixes this per method in its config.
    """

    name: str = "base"
    domain: str = "industrial"
    has_pixel_labels: bool = True  # False for datasets with image-level labels only (e.g. Head CT)

    def __init__(
        self, root: str | Path, categories: list[str] | None = None, image_size: int | None = None
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"{self.name}: dataset root not found: {self.root}")
        self.image_size = image_size
        records = self._scan()
        if categories is not None:
            records = [r for r in records if r.category in set(categories)]
        if not records:
            raise RuntimeError(f"{self.name}: no test images found under {self.root}")
        # Sort so the order (and therefore any sampling with a fixed seed) is reproducible.
        self.records = sorted(records, key=lambda r: r.image_path)

    def _scan(self) -> list[Record]:
        raise NotImplementedError

    @property
    def categories(self) -> list[str]:
        return sorted({r.category for r in self.records})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        w, h = image.size

        if rec.mask_path is not None:
            # Masks come as 0/255 (MVTec, MPDD, BMAD) or 0/k (VisA); anything > 0 is defect.
            mask = (np.array(Image.open(rec.mask_path).convert("L")) > 0).astype(np.uint8)
            if mask.shape != (h, w):  # a few masks are stored at a different size than the image
                mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        if self.image_size is not None:
            s = self.image_size
            image = image.resize((s, s), Image.BICUBIC)
            mask = np.array(Image.fromarray(mask).resize((s, s), Image.NEAREST))

        image_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return {
            "image": image_t,
            "label": int(rec.label),
            "mask": torch.from_numpy(mask),
            "category": rec.category,
            "domain": self.domain,
            "defect_type": rec.defect_type,
            "path": rec.image_path,
            "dataset": self.name,
        }

    def summary(self) -> str:
        """One line per category: number of normal and anomalous test images."""
        lines = [
            f"{self.name} ({self.domain}), {len(self)} test images, pixel labels: {self.has_pixel_labels}"
        ]
        for c in self.categories:
            recs = [r for r in self.records if r.category == c]
            n_anom = sum(r.label for r in recs)
            lines.append(f"  {c:20s} normal={len(recs) - n_anom:5d} anomalous={n_anom:5d}")
        return "\n".join(lines)


def list_images(folder: Path) -> list[Path]:
    """All image files directly inside `folder`, sorted."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
