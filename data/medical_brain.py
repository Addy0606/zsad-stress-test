"""BMAD Brain MRI loader (Bao et al., BMAD benchmark, CVPR-W 2024; slices from BraTS 2021).
Axial FLAIR slices; normal slices vs slices containing a tumour, with pixel masks.
BMAD licence: CC BY-NC-SA 4.0; underlying BraTS 2021 data under its own data-use terms.

Layout (root = .../Brain_AD/BraTS2021_slice):

    root/test/good/img/<id>.png          normal       (some copies have no img/ subfolder)
    root/test/Ungood/img/<id>.png        anomalous
    root/test/Ungood/<mask_dir>/<id>.png mask, same file name; mask_dir is "anomaly_mask"
                                         (BMAD README) or "label" (some re-packaged copies)

Only the test split is used. BMAD also has a `valid` split; we don't use it, so nothing
from it can leak into tuning.
"""

from __future__ import annotations

from pathlib import Path

from data.base import AnomalyDataset, Record, list_images

MASK_DIR_NAMES = ("anomaly_mask", "label")


def _image_dir(folder: Path) -> Path:
    """Images sit either in folder/img/ or directly in folder/."""
    return folder / "img" if (folder / "img").is_dir() else folder


class BMADBrain(AnomalyDataset):
    name = "bmad_brain"
    domain = "medical"

    def _scan(self) -> list[Record]:
        test = self.root / "test"
        good, bad = test / "good", test / "Ungood"
        if not good.is_dir() or not bad.is_dir():
            raise FileNotFoundError(f"expected {good} and {bad}")
        mask_dir = next((bad / d for d in MASK_DIR_NAMES if (bad / d).is_dir()), None)
        if mask_dir is None:
            raise FileNotFoundError(f"no mask folder ({' or '.join(MASK_DIR_NAMES)}) inside {bad}")

        records = [
            Record(image_path=str(p), label=0, category="brain") for p in list_images(_image_dir(good))
        ]
        for p in list_images(_image_dir(bad)):
            mask = mask_dir / p.name
            if not mask.exists():
                raise FileNotFoundError(f"missing mask for {p}: expected {mask}")
            records.append(
                Record(image_path=str(p), label=1, category="brain", defect_type="tumor", mask_path=str(mask))
            )
        return records
