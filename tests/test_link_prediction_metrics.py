"""Tests for astroml.benchmarking.metrics.LinkPredictionMetrics's
ranking metrics (Hits@K, Precision@K, Recall@K).

Issue #841: recall@k empty-pred test.
"""

from __future__ import annotations

import numpy as np
import pytest

from astroml.benchmarking.metrics import LinkPredictionMetrics


@pytest.fixture()
def labelled_scores():
    """5 samples, 3 positives, with scores that rank them unambiguously
    (no ties), so top-k selection is deterministic."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    return y_true, y_scores


class TestRecallAtKEmptyPrediction:
    def test_recall_at_k_zero_is_zero_not_one(self, labelled_scores):
        """The actual bug behind this issue: `np.argsort(y_scores)[-0:]`
        returns the *whole* array (Python/NumPy's `-0 == 0` slicing
        quirk), not an empty one, so recall@0 previously came out as 1.0
        — scored as if every sample had been "retrieved" by an empty
        top-0 prediction set, instead of 0.0."""
        y_true, y_scores = labelled_scores
        assert LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=0) == 0.0

    def test_recall_at_negative_k_is_zero(self, labelled_scores):
        y_true, y_scores = labelled_scores
        assert LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=-5) == 0.0

    def test_recall_at_k_with_no_positives_is_zero_regardless_of_k(self):
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.9, 0.3, 0.7])
        assert LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=0) == 0.0
        assert LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=2) == 0.0

    def test_recall_at_k_with_empty_arrays_is_zero(self):
        y_true = np.array([])
        y_scores = np.array([])
        assert LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=5) == 0.0

    def test_recall_at_a_normal_positive_k_is_unaffected(self, labelled_scores):
        """Regression guard: the fix must not change behavior for the
        common, already-correct case."""
        y_true, y_scores = labelled_scores
        # Top-5 (the whole array) always recalls all 3 positives.
        assert LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=5) == 1.0

    def test_recall_at_k_increases_monotonically_with_k(self, labelled_scores):
        y_true, y_scores = labelled_scores
        values = [LinkPredictionMetrics._recall_at_k(y_true, y_scores, k=k) for k in range(0, 6)]
        assert values == sorted(values)
        assert values[0] == 0.0


class TestPrecisionAtKEmptyPrediction:
    def test_precision_at_k_zero_is_zero_not_a_division_error(self, labelled_scores):
        """The same `[-0:]` bug applied here too, and even a corrected
        empty top-k set would otherwise divide by `k=0` directly."""
        y_true, y_scores = labelled_scores
        assert LinkPredictionMetrics._precision_at_k(y_true, y_scores, k=0) == 0.0

    def test_precision_at_negative_k_is_zero(self, labelled_scores):
        y_true, y_scores = labelled_scores
        assert LinkPredictionMetrics._precision_at_k(y_true, y_scores, k=-3) == 0.0

    def test_precision_at_a_normal_positive_k_is_unaffected(self, labelled_scores):
        y_true, y_scores = labelled_scores
        # Top-2 by score: indices 4 (0.95) and 0 (0.9), both positive.
        assert LinkPredictionMetrics._precision_at_k(y_true, y_scores, k=2) == 1.0


class TestHitsAtKEmptyPrediction:
    def test_hits_at_k_zero_is_zero_not_a_false_hit(self, labelled_scores):
        """An empty top-0 prediction set previously registered a "hit"
        (matching index 0 of the whole array by coincidence), for the
        same root-cause reason as recall@0 and precision@0."""
        y_true, y_scores = labelled_scores
        assert LinkPredictionMetrics._hits_at_k(y_true, y_scores, k=0) == 0.0

    def test_hits_at_negative_k_is_zero(self, labelled_scores):
        y_true, y_scores = labelled_scores
        assert LinkPredictionMetrics._hits_at_k(y_true, y_scores, k=-1) == 0.0

    def test_hits_at_k_with_empty_y_true_is_zero_not_a_division_error(self):
        """`hits / len(y_true)` divides by the length of `y_true`
        directly; an empty `y_true` must not raise ZeroDivisionError."""
        y_true = np.array([])
        y_scores = np.array([])
        assert LinkPredictionMetrics._hits_at_k(y_true, y_scores, k=5) == 0.0

    def test_hits_at_a_normal_positive_k_is_unaffected(self, labelled_scores):
        y_true, y_scores = labelled_scores
        # Top-2 includes a positive (index 4), 1 hit out of 5 samples.
        assert LinkPredictionMetrics._hits_at_k(y_true, y_scores, k=2) == pytest.approx(0.2)


class TestComputeWithKZero:
    def test_compute_reports_zero_for_every_at_k_metric_when_k_zero_is_requested(
        self, labelled_scores
    ):
        """End-to-end: `LinkPredictionMetrics.compute` with `k_values=[0]`
        must surface the corrected 0.0 for all three ranking metrics, not
        just the private helpers tested in isolation above."""
        y_true, y_scores = labelled_scores
        y_pred = (y_scores >= 0.5).astype(int)

        metrics = LinkPredictionMetrics.compute(y_true, y_pred, y_prob=y_scores, k_values=[0])

        assert metrics["hits_at_0"] == 0.0
        assert metrics["precision_at_0"] == 0.0
        assert metrics["recall_at_0"] == 0.0
