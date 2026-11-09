"""VisA loader (Zou et al., ECCV 2022). 12 categories, 2,162 test images. CC BY 4.0.

Uses the official one-class split file and the per-category annotation files:

    root/split_csv/1cls.csv              object,split,label,image,mask  (paths relative to root)
    root/<category>/image_anno.csv       image,label,mask  (label = defect description(s))
"""

from __future__ import annotations

import csv

from data.base import AnomalyDataset, Record


class VisA(AnomalyDataset):
    name = "visa"
    domain = "industrial"

    def _defect_types(self, category: str) -> dict[str, str]:
        """Map image path -> defect description, read from <category>/image_anno.csv."""
        anno = self.root / category / "image_anno.csv"
        if not anno.exists():
            return {}
        with open(anno, newline="", encoding="utf-8") as fh:
            # Some images have several defects, stored as one comma-separated string.
            return {row["image"]: row["label"].strip() for row in csv.DictReader(fh)}

    def _scan(self) -> list[Record]:
        split_file = self.root / "split_csv" / "1cls.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"VisA split file not found: {split_file}")
        defect_maps: dict[str, dict[str, str]] = {}
        records = []
        with open(split_file, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["split"] != "test":
                    continue
                cat = row["object"]
                is_anomaly = row["label"] == "anomaly"
                if cat not in defect_maps:
                    defect_maps[cat] = self._defect_types(cat)
                defect = defect_maps[cat].get(row["image"], "unknown") if is_anomaly else None
                records.append(
                    Record(
                        image_path=str(self.root / row["image"]),
                        label=int(is_anomaly),
                        category=cat,
                        defect_type=defect,
                        mask_path=str(self.root / row["mask"]) if is_anomaly and row["mask"] else None,
                    )
                )
        return records
