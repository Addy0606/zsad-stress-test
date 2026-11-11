"""Leave-one-dataset-out splits for the fusion heads (Phase 6).

Rules (docs/brief.md, section 5):
- Folds are the three datasets with pixel masks: MVTec AD, VisA, BMAD Brain MRI.
  For each, a head is trained on the other two and tested on it.
- MPDD and Head CT are NEVER used for training (or validation). They are scored by a head trained on
  all three fold datasets, in the role the spec gave the held-out custom dataset.
- Hyperparameters are chosen only on the training datasets, by an inner leave-one-dataset-out loop:
  train on all training datasets but one, validate on that one, average over the choices.
"""

from __future__ import annotations

from dataclasses import dataclass

FOLD_DATASETS = ("mvtec", "visa", "bmad_brain")
TEST_ONLY_DATASETS = ("mpdd", "headct")


@dataclass(frozen=True)
class Split:
    test: str
    train: tuple[str, ...]

    def inner(self) -> list[tuple[tuple[str, ...], str]]:
        """(train, validation) pairs for choosing hyperparameters, using the training datasets only."""
        return [(tuple(d for d in self.train if d != v), v) for v in self.train]


def outer_splits() -> list[Split]:
    """One split per evaluated dataset; the test dataset is never among the training datasets."""
    splits = [Split(test=d, train=tuple(x for x in FOLD_DATASETS if x != d)) for d in FOLD_DATASETS]
    splits += [Split(test=d, train=FOLD_DATASETS) for d in TEST_ONLY_DATASETS]
    for s in splits:
        assert s.test not in s.train, s
        assert not set(s.train) & set(TEST_ONLY_DATASETS), s
    return splits
