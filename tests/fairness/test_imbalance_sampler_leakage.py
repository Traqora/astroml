"""Leakage regression tests for imbalance-aware sampling (issue #886).

Resampling is a common fix for class/group imbalance, and an equally common
source of train/test leakage: a sampler run over a combined pool can copy a
held-out row into the training set, or a per-fold splitter can overlap its
folds. These tests pin the two properties that make the samplers in this repo
safe to use:

* every row :meth:`BiasMitigation.sampling` returns comes from the input it
  was given — oversampling duplicates, it never invents a row; and
* :class:`StratifiedKFoldSplitter` partitions the data into disjoint,
  fully-covering folds while preserving each class's proportion.
"""

from __future__ import annotations

import numpy as np
import pytest

from astroml.validation.fairness.mitigation import BiasMitigation
from astroml.validation.splitters import GroupKFoldSplitter, StratifiedKFoldSplitter


@pytest.fixture
def mitigator() -> BiasMitigation:
    return BiasMitigation()


@pytest.fixture
def imbalanced_rows():
    """100 rows: 80 majority (group 0) and 20 minority (group 1)."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = np.array([0] * 80 + [1] * 20)
    sensitive = np.array([0] * 80 + [1] * 20)
    return X, y, sensitive


class TestImbalanceSamplerProvenance:
    """The resampler must never introduce a row that was not in its input."""

    def test_undersampling_only_selects_original_rows(self, mitigator, imbalanced_rows):
        X, y, sensitive = imbalanced_rows
        X_res, y_res, sf_res = mitigator.sampling(
            X, y, sensitive, strategy="undersampling", random_state=7
        )

        original_ids = set(X.ravel().tolist())
        assert set(X_res.ravel().tolist()) <= original_ids
        # Rows are drawn without replacement, so the result has no duplicates.
        assert len(set(X_res.ravel().tolist())) == len(X_res)

    def test_oversampling_duplicates_but_never_fabricates(self, mitigator, imbalanced_rows):
        X, y, sensitive = imbalanced_rows
        X_res, _, _ = mitigator.sampling(X, y, sensitive, strategy="oversampling", random_state=7)

        original_ids = set(X.ravel().tolist())
        assert set(X_res.ravel().tolist()) <= original_ids
        # Oversampling grows the minority class by repetition.
        assert len(X_res) > len(X)

    @pytest.mark.parametrize("strategy", ["undersampling", "oversampling"])
    def test_sampled_groups_are_balanced(self, mitigator, imbalanced_rows, strategy):
        X, y, sensitive = imbalanced_rows
        _, _, sf_res = mitigator.sampling(X, y, sensitive, strategy=strategy, random_state=7)
        counts = np.bincount(sf_res)
        assert counts.min() == counts.max()

    def test_same_seed_reproduces_the_sample(self, mitigator, imbalanced_rows):
        X, y, sensitive = imbalanced_rows
        first = mitigator.sampling(X, y, sensitive, strategy="undersampling", random_state=3)
        second = mitigator.sampling(X, y, sensitive, strategy="undersampling", random_state=3)
        assert np.array_equal(first[0], second[0])

    def test_held_out_rows_do_not_leak_into_training(self, mitigator, imbalanced_rows):
        """Fit the sampler on train only; test rows must not appear in the output."""
        X, y, sensitive = imbalanced_rows
        train_idx = np.arange(0, 80)
        test_idx = np.arange(80, 100)

        X_train, y_train, sf_train = X[train_idx], y[train_idx], sensitive[train_idx]
        X_res, _, _ = mitigator.sampling(
            X_train, y_train, sf_train, strategy="oversampling", random_state=1
        )

        train_ids = set(X_train.ravel().tolist())
        test_ids = set(X[test_idx].ravel().tolist())
        assert set(X_res.ravel().tolist()) <= train_ids
        assert train_ids.isdisjoint(test_ids)


class TestStratifiedFoldLeakage:
    """The imbalanced-aware splitter must not let a row cross folds."""

    def test_folds_are_disjoint_and_cover_every_row(self):
        y = np.array([0] * 80 + [1] * 20)
        X = np.arange(100, dtype=float).reshape(-1, 1)
        splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=42)

        all_test = []
        for train_idx, test_idx in splitter.split(X, y):
            assert set(train_idx).isdisjoint(test_idx)
            all_test.append(test_idx)

        # Every sample is tested exactly once and none is dropped.
        combined = np.concatenate(all_test)
        assert sorted(combined.tolist()) == list(range(100))

    def test_minority_class_is_present_in_every_fold(self):
        y = np.array([0] * 80 + [1] * 20)
        X = np.arange(100, dtype=float).reshape(-1, 1)
        splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=1)

        for _, test_idx in splitter.split(X, y):
            assert np.sum(y[test_idx] == 1) > 0

    def test_class_proportion_is_preserved_per_fold(self):
        y = np.array([0] * 80 + [1] * 20)
        X = np.arange(100, dtype=float).reshape(-1, 1)
        splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)

        for _, test_idx in splitter.split(X, y):
            assert np.sum(y[test_idx] == 1) == 4
            assert np.sum(y[test_idx] == 0) == 16

    def test_split_is_reproducible_for_a_fixed_seed(self):
        y = np.array([0] * 50 + [1] * 10)
        X = np.arange(60, dtype=float).reshape(-1, 1)
        s1 = StratifiedKFoldSplitter(n_splits=3, shuffle=True, random_state=5)
        s2 = StratifiedKFoldSplitter(n_splits=3, shuffle=True, random_state=5)

        for (tr1, te1), (tr2, te2) in zip(s1.split(X, y), s2.split(X, y)):
            assert np.array_equal(tr1, tr2)
            assert np.array_equal(te1, te2)


class TestGroupFoldLeakage:
    """Group folds must keep a group entirely on one side of the split."""

    def test_no_group_is_shared_between_train_and_test(self):
        groups = np.repeat(np.arange(10), 10)
        X = np.arange(100, dtype=float).reshape(-1, 1)
        splitter = GroupKFoldSplitter(n_splits=5)

        for train_idx, test_idx in splitter.split(X, groups=groups):
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
