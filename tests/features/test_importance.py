"""Tests for feature importance and selection utilities (#744)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astroml.features.importance import (
    FeatureImportanceReport,
    compute_permutation_importance,
    default_selection_filters,
    select_features,
)


def _linear_frame(n=120, seed=7):
    """X with a strong signal column, a weak one, and a useless one."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    weak = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    X = pd.DataFrame({"signal": signal, "weak": weak, "noise": noise})
    y = 3.0 * signal + 0.3 * weak + rng.normal(0, 0.1, n)
    return X, np.asarray(y)


def _predict(X):
    """A deterministic stand-in for a fitted model."""
    weights = np.array([3.0, 0.3, 0.0])
    return X.to_numpy() @ weights


class TestPermutationImportance:
    """Permutation-importance computation."""

    def test_signal_feature_dominates(self):
        X, y = _linear_frame()
        report = compute_permutation_importance(_predict, X, y, n_repeats=5, random_state=42)
        assert isinstance(report, FeatureImportanceReport)
        ranked = report.ranked()
        assert ranked[0].feature == "signal"
        assert ranked[-1].feature == "noise"
        assert report.top_k(1)[0].feature == "signal"

    def test_baseline_matches_unpermuted_score(self):
        X, y = _linear_frame()
        report = compute_permutation_importance(_predict, X, y, n_repeats=3, random_state=0)
        expected = 1.0 - float(np.sum((y - _predict(X)) ** 2) / np.sum((y - y.mean()) ** 2))
        assert report.baseline_score == pytest.approx(expected, abs=1e-9)

    def test_reproducible_with_seed(self):
        X, y = _linear_frame()
        a = compute_permutation_importance(_predict, X, y, n_repeats=3, random_state=11)
        b = compute_permutation_importance(_predict, X, y, n_repeats=3, random_state=11)
        assert [i.importance for i in a.importances] == [i.importance for i in b.importances]

    def test_feature_subset(self):
        X, y = _linear_frame()
        report = compute_permutation_importance(_predict, X, y, features=["signal"], n_repeats=2)
        assert [i.feature for i in report.importances] == ["signal"]

    def test_missing_feature_raises(self):
        X, y = _linear_frame()
        with pytest.raises(KeyError, match="missing_feature"):
            compute_permutation_importance(_predict, X, y, features=["missing_feature"])

    def test_invalid_repeats_raises(self):
        X, y = _linear_frame()
        with pytest.raises(ValueError, match="n_repeats"):
            compute_permutation_importance(_predict, X, y, n_repeats=0)

    def test_to_frame_sorted_descending(self):
        X, y = _linear_frame()
        report = compute_permutation_importance(_predict, X, y, n_repeats=3)
        frame = report.to_frame()
        assert list(frame["importance"]) == sorted(frame["importance"], reverse=True)
        assert list(frame.columns) == [
            "feature",
            "importance",
            "permuted_score_mean",
            "permuted_score_std",
        ]

    def test_binary_target_uses_accuracy(self):
        rng = np.random.default_rng(3)
        n = 80
        X = pd.DataFrame({"signal": rng.normal(0, 1, n), "noise": rng.normal(0, 1, n)})
        y = (X["signal"] > 0).astype(int).to_numpy()
        report = compute_permutation_importance(
            lambda frame: (frame["signal"] > 0).astype(int).to_numpy(),
            X,
            y,
            n_repeats=3,
        )
        assert report.scorer_name == "_accuracy"
        assert report.ranked()[0].feature == "signal"


class TestSelectionFilters:
    """Static filters: variance, missing rate, correlation."""

    def test_variance_filter_drops_constants(self):
        X = pd.DataFrame({"const": [1.0] * 50, "vary": np.arange(50, dtype=float)})
        filters = default_selection_filters(variance_threshold=0.01)
        reasons = filters[0](X)
        assert "const" in reasons
        assert "vary" not in reasons

    def test_missing_filter_drops_sparse_columns(self):
        X = pd.DataFrame(
            {
                "sparse": [np.nan] * 5 + list(range(5)),
                "dense": list(range(10)),
            }
        )
        filters = default_selection_filters(max_missing_rate=0.3)
        reasons = filters[1](X)
        assert "sparse" in reasons
        assert "dense" not in reasons

    def test_correlation_filter_drops_collinear_pair(self):
        rng = np.random.default_rng(5)
        base = np.arange(50, dtype=float)
        X = pd.DataFrame(
            {
                "a": base,
                "copy": base * 2.0,
                "other": base[::-1] + rng.normal(0, 10, 50),  # weakly related
            }
        )
        filters = default_selection_filters(correlation_threshold=0.95)
        reasons = filters[2](X)
        dropped = set(reasons)
        # One of the collinear pair is dropped, never both, never 'other'.
        assert ("a" in dropped) ^ ("copy" in dropped)
        assert "other" not in dropped


class TestSelectFeatures:
    """End-to-end selection behaviour."""

    def test_empty_frame(self):
        report = select_features(pd.DataFrame())
        assert report.selected == []

    def test_filters_only(self):
        X = pd.DataFrame(
            {
                "good": np.arange(50, dtype=float),
                "const": [7.0] * 50,
                "collinear": np.arange(50, dtype=float) * 1.0,
            }
        )
        report = select_features(X, filters=default_selection_filters(correlation_threshold=0.95))
        assert "const" not in report.selected
        assert "good" in report.selected
        assert report.dropped

    def test_top_k_with_predict(self):
        X, y = _linear_frame()
        report = select_features(X, y, predict=_predict, top_k=1)
        assert report.selected == ["signal"]
        assert "noise" in report.dropped

    def test_importance_threshold(self):
        X, y = _linear_frame()
        report = select_features(X, y, predict=_predict, importance_threshold=0.1, top_k=None)
        assert "signal" in report.selected
        assert "noise" not in report.selected

    def test_transform_reduces_columns(self):
        X, y = _linear_frame()
        report = select_features(X, y, predict=_predict, top_k=2)
        reduced = report.transform(X)
        assert list(reduced.columns) == report.selected
        assert len(reduced) == len(X)

    def test_predict_without_y_raises(self):
        X, _ = _linear_frame()
        with pytest.raises(ValueError, match="y is required"):
            select_features(X, None, predict=_predict)

    def test_importance_options_ignored_without_predict(self):
        rng = np.random.default_rng(6)
        X = pd.DataFrame(
            {
                "a": np.arange(30, dtype=float),
                "b": np.arange(30, dtype=float) * 2.0,
                "noise": rng.normal(0, 25, 30),
            }
        )
        report = select_features(X, top_k=1, importance_threshold=0.5)
        # Without a predictor there are no importance scores to cut on; the
        # correlation filter still removes one of the collinear columns.
        assert report.importances == {}
        assert report.selected
