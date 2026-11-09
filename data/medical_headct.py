"""Head CT (hemorrhage) loader. 200 CT slices: 100 normal, 100 with hemorrhage. CC0.
Source: kaggle.com/datasets/felipekitamura/head-ct-hemorrhage

    root/labels.csv               id, hemorrhage   (id 0 -> 000.png; hemorrhage 1 = anomalous)
    root/head_ct/head_ct/000.png

There are **no pixel masks**, so this dataset only supports image-level metrics.
Masks are returned as all zeros to keep the interface uniform; `has_pixel_labels = False`
tells the evaluation code to skip pixel metrics.
"""

from __future__ import annotations

import csv

from data.base import AnomalyDataset, Record


class HeadCT(AnomalyDataset):
    name = "headct"
    domain = "medical"
    has_pixel_labels = False

    def _scan(self) -> list[Record]:
        labels = self.root / "labels.csv"
        if not labels.exists():
            raise FileNotFoundError(f"Head CT labels not found: {labels}")
        img_dir = self.root / "head_ct" / "head_ct"
        records = []
        with open(labels, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)  # header: "id, hemorrhage" (note the space)
            for row in reader:
                idx, hemorrhage = int(row[0]), int(row[1])
                records.append(
                    Record(
                        image_path=str(img_dir / f"{idx:03d}.png"),
                        label=hemorrhage,
                        category="head_ct",
                        defect_type="hemorrhage" if hemorrhage else None,
                        mask_path=None,
                    )
                )
        return records
