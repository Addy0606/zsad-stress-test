"""Tests for the fusion split rules and the train-only normaliser (Phase 6).

The spec requires a test that the fusion split never leaks a held-out dataset into training."""

from __future__ import annotations

import numpy as np
import pytest

from fusion.normalize import QuantileNormalizer
from fusion.splits import FOLD_DATASETS, TEST_ONLY_DATASETS, outer_splits


def test_every_dataset_is_held_out_exactly_once():
    tests = [s.test for s in outer_splits()]
    assert sorted(tests) == sorted(FOLD_DATASETS + TEST_ONLY_DATASETS)


def test_test_dataset_never_in_training_or_validation():
    for s in outer_splits():
        assert s.test not in s.train
        for inner_train, val in s.inner():
            assert s.test not in inner_train and s.test != val
            assert val not in inner_train


def test_test_only_datasets_never_train():
    for s in outer_splits():
        assert not set(s.train) & set(TEST_ONLY_DATASETS)
        for inner_train, val in s.inner():
            assert not set(inner_train) & set(TEST_ONLY_DATASETS) and val not in TEST_ONLY_DATASETS


def test_fold_splits_train_on_the_other_two():
    for s in outer_splits():
        if s.test in FOLD_DATASETS:
            assert set(s.train) == set(FOLD_DATASETS) - {s.test}
        else:
            assert set(s.train) == set(FOLD_DATASETS)


def test_quantile_normalizer_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    train = np.c_[rng.normal(size=500), rng.exponential(size=500)]
    norm = QuantileNormalizer().fit(train)
    x = np.c_[np.linspace(-5, 5, 101), np.linspace(-1, 10, 101)]
    z = norm.transform(x)
    assert (np.diff(z, axis=0) >= 0).all()
    assert z.min() >= 0 and z.max() <= 1
    assert norm.transform(np.array([[0.0, np.median(train[:, 1])]]))[0] == pytest.approx([0.5, 0.5], abs=0.03)


def test_quantile_normalizer_handles_ties():
    train = np.r_[np.zeros(300), np.ones(200)][:, None]  # heavy ties
    z = QuantileNormalizer().fit(train).transform(np.array([[0.0], [0.5], [1.0]]))
    assert 0 <= z[0, 0] < z[1, 0] < z[2, 0] <= 1


def test_normalizer_depends_only_on_fit_data():
    """Transforming different test sets with the same fitted normaliser gives the same values for the same inputs."""
    rng = np.random.default_rng(1)
    norm = QuantileNormalizer().fit(rng.normal(size=(400, 2)))
    a = norm.transform(np.array([[0.3, -1.0]]))
    norm.transform(rng.normal(loc=5, size=(1000, 2)))  # seeing other data must not change anything
    np.testing.assert_array_equal(a, norm.transform(np.array([[0.3, -1.0]])))
