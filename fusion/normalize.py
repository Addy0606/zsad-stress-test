"""Put every feature on a common scale using the TRAINING datasets only.

Each detector's raw scores live on their own scale (AnomalyCLIP gives probabilities near 1, AnomalyVFM gives
sigmoid outputs, map values differ again). A fusion head needs them comparable. We map each feature through
its empirical distribution on the training data (a quantile transform): a value becomes "the fraction of
training values below it", in [0, 1]. This is monotone, so it never changes a single feature's AUROC, and it
is fitted without looking at the test dataset.

Rescaling with the test images' own statistics would be simpler, but it would let the head use the unlabelled
test set (transductive), which single-image methods can't do.
"""

from __future__ import annotations

import numpy as np

N_QUANTILES = 1001


class QuantileNormalizer:
    def __init__(self, n_quantiles: int = N_QUANTILES) -> None:
        self.n_quantiles = n_quantiles
        self.grid_: np.ndarray | None = None  # (n_quantiles, n_features)

    def fit(self, train: np.ndarray) -> QuantileNormalizer:
        """train: (n_samples, n_features), from the training datasets only."""
        probs = np.linspace(0, 1, self.n_quantiles)
        self.grid_ = np.quantile(np.asarray(train, dtype=np.float64), probs, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Each column mapped to [0, 1] by interpolating within its training quantiles."""
        assert self.grid_ is not None, "fit first"
        x = np.asarray(x, dtype=np.float64)
        probs = np.linspace(0, 1, self.n_quantiles)
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            grid = self.grid_[:, j]
            # equal quantiles (ties) would make interp ill-defined: average the probabilities of tied values
            uniq, inv = np.unique(grid, return_inverse=True)
            p = np.bincount(inv, weights=probs) / np.bincount(inv)
            out[:, j] = np.interp(x[:, j], uniq, p, left=0.0, right=1.0)
        return out
