"""Tests for FairnessMetrics."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.validation.fairness.metrics import (
    DEMOGRAPHIC_PARITY_THRESHOLD,
    DISPARATE_IMPACT_LOWER,
    DISPARATE_IMPACT_UPPER,
    EQUAL_OPPORTUNITY_THRESHOLD,
    EQUALIZED_ODDS_THRESHOLD,
    FairnessMetricResult,
    FairnessMetrics,
)


@pytest.fixture
def metrics() -> FairnessMetrics:
    return FairnessMetrics()


class TestFairnessMetrics:
    def test_demographic_parity_fair(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = metrics.demographic_parity(y_true, y_pred, sensitive)
        assert isinstance(result, FairnessMetricResult)
        assert result.metric_name == "demographic_parity"
        assert isinstance(result.passed, bool)
        assert len(result.group_metrics) > 0

    def test_demographic_parity_biased(self, metrics, biased_dataset):
        y_true, y_pred, sensitive = biased_dataset
        result = metrics.demographic_parity(y_true, y_pred, sensitive)
        assert result.metric_name == "demographic_parity"
        assert result.value >= 0

    def test_equal_opportunity_fair(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = metrics.equal_opportunity(y_true, y_pred, sensitive)
        assert result.metric_name == "equal_opportunity"
        assert isinstance(result.passed, bool)

    def test_equal_opportunity_positive_class(self, metrics):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array([0, 0, 1, 1, 0, 1])
        result = metrics.equal_opportunity(y_true, y_pred, sensitive, positive_class=1)
        assert result.metric_name == "equal_opportunity"

    def test_equalized_odds(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = metrics.equalized_odds(y_true, y_pred, sensitive)
        assert result.metric_name == "equalized_odds"
        assert isinstance(result.passed, bool)

    def test_disparate_impact(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = metrics.disparate_impact(y_true, y_pred, sensitive)
        assert result.metric_name == "disparate_impact"
        assert result.value >= 0

    def test_disparate_impact_zero_denominator(self, metrics):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        sensitive = np.array([0, 0, 1, 1])
        result = metrics.disparate_impact(y_true, y_pred, sensitive)
        assert result.value == float("inf")
        assert not result.passed

    def test_statistical_parity_alias(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        dp = metrics.demographic_parity(y_true, y_pred, sensitive)
        sp = metrics.statistical_parity(y_true, y_pred, sensitive)
        assert dp.value == sp.value
        assert dp.passed == sp.passed

    def test_compute_all(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        results = metrics.compute_all(y_true, y_pred, sensitive)
        assert len(results) == 4
        assert "demographic_parity" in results
        assert "equal_opportunity" in results
        assert "equalized_odds" in results
        assert "disparate_impact" in results
        for name, result in results.items():
            assert isinstance(result, FairnessMetricResult)
            assert result.metric_name == name

    def test_empty_data_raises(self, metrics):
        with pytest.raises(ValueError, match="empty"):
            metrics.demographic_parity(np.array([]), np.array([]), np.array([]))

    def test_shape_mismatch_raises(self, metrics):
        with pytest.raises(ValueError, match="shape"):
            metrics.demographic_parity(np.array([1, 0]), np.array([1]), np.array([0, 1]))

    def test_non_binary_labels_raises(self, metrics):
        with pytest.raises(ValueError, match="binary"):
            metrics.demographic_parity(
                np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([0, 0, 1])
            )

    def test_non_binary_preds_raises(self, metrics):
        with pytest.raises(ValueError, match="binary"):
            metrics.demographic_parity(
                np.array([0, 1, 0]), np.array([0, 1, 2]), np.array([0, 0, 1])
            )

    def test_threshold_values(self, metrics):
        assert DEMOGRAPHIC_PARITY_THRESHOLD == 0.1
        assert EQUAL_OPPORTUNITY_THRESHOLD == 0.1
        assert EQUALIZED_ODDS_THRESHOLD == 0.1
        assert DISPARATE_IMPACT_LOWER == 0.8
        assert DISPARATE_IMPACT_UPPER == 1.25

    def test_group_metrics_populated(self, metrics):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array([0, 0, 1, 1, 0, 1])
        result = metrics.demographic_parity(y_true, y_pred, sensitive)
        assert "0" in result.group_metrics
        assert "1" in result.group_metrics

    def test_biased_dataset_fails_parity(self, metrics, biased_dataset_unequal):
        y_true, y_pred, sensitive = biased_dataset_unequal
        dp = metrics.demographic_parity(y_true, y_pred, sensitive)
        eo = metrics.equal_opportunity(y_true, y_pred, sensitive)
        assert not dp.passed or not eo.passed or not dp.passed

    def test_three_groups(self, metrics):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
        sensitive = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        result = metrics.demographic_parity(y_true, y_pred, sensitive)
        assert len(result.group_metrics) == 3

    def test_metric_result_details(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = metrics.demographic_parity(y_true, y_pred, sensitive)
        assert result.details is not None
        assert "Demographic parity" in result.details

    def test_threshold_tuple_for_disparate_impact(self, metrics, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = metrics.disparate_impact(y_true, y_pred, sensitive)
        assert isinstance(result.threshold, tuple)
        assert len(result.threshold) == 2
