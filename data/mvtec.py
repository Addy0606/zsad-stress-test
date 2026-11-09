"""MVTec AD and MPDD loaders. Both use the same folder layout:

    root/<category>/test/good/000.png                      normal
    root/<category>/test/<defect_type>/000.png             anomalous
    root/<category>/ground_truth/<defect_type>/000_mask.png

Only the test split is used: zero-shot methods never see the train split.
"""

from __future__ import annotations

from data.base import AnomalyDataset, Record, list_images


class MVTecLayoutDataset(AnomalyDataset):
    """Any dataset stored in the MVTec AD folder layout."""

    def _scan(self) -> list[Record]:
        records = []
        for cat_dir in sorted(p for p in self.root.iterdir() if (p / "test").is_dir()):
            for defect_dir in sorted(p for p in (cat_dir / "test").iterdir() if p.is_dir()):
                is_normal = defect_dir.name == "good"
                for img in list_images(defect_dir):
                    mask = None
                    if not is_normal:
                        mask_path = cat_dir / "ground_truth" / defect_dir.name / f"{img.stem}_mask.png"
                        if not mask_path.exists():
                            raise FileNotFoundError(f"missing mask for {img}: expected {mask_path}")
                        mask = str(mask_path)
                    records.append(
                        Record(
                            image_path=str(img),
                            label=0 if is_normal else 1,
                            category=cat_dir.name,
                            defect_type=None if is_normal else defect_dir.name,
                            mask_path=mask,
                        )
                    )
        return records


class MVTecAD(MVTecLayoutDataset):
    """MVTec AD (Bergmann et al., CVPR 2019). 15 categories, 1,725 test images. CC BY-NC-SA 4.0."""

    name = "mvtec"
    domain = "industrial"


class MPDD(MVTecLayoutDataset):
    """MPDD, Metal Parts Defect Detection (Jezek et al., ICUMT 2021). 6 categories, 458 test images.
    CC BY-NC-SA 4.0. Test-only stress set in this project (never used for tuning or fusion training)."""

    name = "mpdd"
    domain = "industrial"
