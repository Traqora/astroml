"""Per-cohort fairness metric tests (#901)."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.validation.fairness.metrics import FairnessMetrics


@pytest.fixture
def metrics() -> FairnessMetrics:
    return FairnessMetrics()


def test_per_cohort_rates_are_exact(metrics):
    # cohort a: 3/4 predicted positive, cohort b: 1/4, cohort c: 2/4
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
    y_true = y_pred.copy()
    cohorts = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    dp = metrics.demographic_parity(y_true, y_pred, cohorts)
    assert dp.group_metrics == pytest.approx({"a": 0.75, "b": 0.25, "c": 0.5})
    assert dp.value == pytest.approx(0.5)
    di = metrics.disparate_impact(y_true, y_pred, cohorts)
    assert di.group_metrics == dp.group_metrics
    assert di.value == pytest.approx(3.0)


def test_per_cohort_tpr_and_fpr(metrics):
    #            cohort 0             cohort 1
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    cohorts = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    eo = metrics.equal_opportunity(y_true, y_pred, cohorts)
    assert eo.group_metrics == pytest.approx({"0": 1.0, "1": 0.5})
    assert eo.value == pytest.approx(0.5)
    odds = metrics.equalized_odds(y_true, y_pred, cohorts)
    # TPR gap 0.5, FPR gap 0.5
    assert odds.value == pytest.approx(0.5)
    assert odds.group_metrics == pytest.approx({"0": 1.0, "1": 0.5})


def test_cohort_without_positives_does_not_inflate_equal_opportunity(metrics):
    """Regression: a cohort with no positive labels used to count as TPR 0."""
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0])
    cohorts = np.array([0, 0, 1, 1, 2, 2])  # cohort 2 has no positives
    eo = metrics.equal_opportunity(y_true, y_pred, cohorts)
    assert eo.value == pytest.approx(0.0)
    assert eo.passed
    assert set(eo.group_metrics) == {"0", "1", "2"}


def test_cohort_without_negatives_does_not_inflate_equalized_odds(metrics):
    y_true = np.array([1, 0, 1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1])
    cohorts = np.array([0, 0, 1, 1, 2, 2])  # cohort 2 has no negatives
    odds = metrics.equalized_odds(y_true, y_pred, cohorts)
    assert odds.value == pytest.approx(0.0)
    assert odds.passed


def test_single_defined_cohort_has_zero_gap(metrics):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    cohorts = np.array([0, 0, 1, 1])  # only cohort 0 has positives
    eo = metrics.equal_opportunity(y_true, y_pred, cohorts)
    assert eo.value == 0.0


def test_singleton_cohort_is_reported(metrics):
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0])
    cohorts = np.array(["big", "big", "big", "big", "solo"])
    dp = metrics.demographic_parity(y_true, y_pred, cohorts)
    assert dp.group_metrics == pytest.approx({"big": 0.5, "solo": 0.0})


def test_cohort_order_does_not_change_results(metrics):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 60)
    y_pred = rng.integers(0, 2, 60)
    cohorts = rng.integers(0, 3, 60)
    perm = rng.permutation(60)
    a = metrics.compute_all(y_true, y_pred, cohorts)
    b = metrics.compute_all(y_true[perm], y_pred[perm], cohorts[perm])
    for name in a:
        assert a[name].value == pytest.approx(b[name].value)
        assert a[name].group_metrics == pytest.approx(b[name].group_metrics)
