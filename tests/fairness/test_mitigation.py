"""Tests for BiasMitigation."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.validation.fairness.mitigation import BiasMitigation, MitigationResult


@pytest.fixture
def mitigator() -> BiasMitigation:
    return BiasMitigation()


class TestBiasMitigation:
    def test_reweighing(self, mitigator, sample_weights_fixture):
        X, y, sensitive = sample_weights_fixture
        weights = mitigator.reweighing(X, y, sensitive)
        assert isinstance(weights, np.ndarray)
        assert weights.shape == y.shape
        assert np.all(weights > 0)

    def test_reweighing_unbalanced(self, mitigator):
        X = np.random.default_rng(42).normal(size=(50, 2))
        y = np.array([1] * 40 + [0] * 10)
        sensitive = np.array([0] * 25 + [1] * 25)
        weights = mitigator.reweighing(X, y, sensitive)
        assert np.all(weights > 0)
        assert not np.allclose(weights, 1.0)

    def test_sampling_undersampling(self, mitigator):
        X = np.random.default_rng(42).normal(size=(100, 3))
        y = np.random.default_rng(42).integers(0, 2, size=100)
        sensitive = np.array([0] * 70 + [1] * 30)
        X_res, y_res, sf_res = mitigator.sampling(X, y, sensitive, strategy="undersampling")
        assert len(X_res) <= len(X)
        assert len(y_res) == len(X_res)
        assert len(sf_res) == len(X_res)

    def test_sampling_oversampling(self, mitigator):
        X = np.random.default_rng(42).normal(size=(100, 3))
        y = np.random.default_rng(42).integers(0, 2, size=100)
        sensitive = np.array([0] * 70 + [1] * 30)
        X_res, y_res, sf_res = mitigator.sampling(X, y, sensitive, strategy="oversampling")
        assert len(X_res) >= len(X)

    def test_sampling_invalid_strategy(self, mitigator):
        X = np.random.default_rng(42).normal(size=(10, 2))
        y = np.random.default_rng(42).integers(0, 2, size=10)
        sensitive = np.random.default_rng(42).integers(0, 2, size=10)
        with pytest.raises(ValueError, match="Unknown strategy"):
            mitigator.sampling(X, y, sensitive, strategy="invalid")

    def test_adversarial_debiasing(self, mitigator, simple_classifier):
        X = np.random.default_rng(42).normal(size=(50, 3))
        y = np.random.default_rng(42).integers(0, 2, size=50)
        sensitive = np.random.default_rng(42).integers(0, 2, size=50)
        history = mitigator.adversarial_debiasing(simple_classifier, X, y, sensitive, epochs=3)
        assert "loss" in history
        assert "adversary_loss" in history
        assert len(history["loss"]) == 3

    def test_equalized_odds_postprocessing(self, mitigator):
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.uniform(0, 1, size=n)
        sensitive = rng.integers(0, 2, size=n)
        adjusted = mitigator.equalized_odds_postprocessing(y_pred, sensitive, y_true)
        assert adjusted.shape == y_pred.shape
        assert set(np.unique(adjusted)).issubset({0.0, 1.0})

    def test_reject_option_classification(self, mitigator):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=(n, 3))
        y_pred = rng.uniform(0, 1, size=n)
        y_true = rng.integers(0, 2, size=n)
        sensitive = np.array([0] * 50 + [1] * 50)
        adjusted = mitigator.reject_option_classification(X, y_pred, y_true, sensitive)
        assert adjusted.shape == y_pred.shape

    def test_mitigate_reweighing(self, mitigator):
        X = np.random.default_rng(42).normal(size=(50, 3))
        y = np.random.default_rng(42).integers(0, 2, size=50)
        sensitive = np.random.default_rng(42).integers(0, 2, size=50)
        result = mitigator.mitigate(X, y, sensitive, strategy="reweighing")
        assert isinstance(result, MitigationResult)
        assert result.strategy_used == "reweighing"
        assert "demographic_parity" in result.before_metrics
        assert isinstance(result.improvement, dict)

    def test_mitigate_sampling(self, mitigator):
        X = np.random.default_rng(42).normal(size=(50, 3))
        y = np.random.default_rng(42).integers(0, 2, size=50)
        sensitive = np.random.default_rng(42).integers(0, 2, size=50)
        result = mitigator.mitigate(
            X, y, sensitive, strategy="sampling", sampling_strategy="undersampling"
        )
        assert result.strategy_used == "sampling"

    def test_mitigate_adversarial(self, mitigator, simple_classifier):
        X = np.random.default_rng(42).normal(size=(30, 3))
        y = np.random.default_rng(42).integers(0, 2, size=30)
        sensitive = np.random.default_rng(42).integers(0, 2, size=30)
        result = mitigator.mitigate(
            X, y, sensitive, strategy="adversarial", model=simple_classifier, epochs=2
        )
        assert result.strategy_used == "adversarial"

    def test_mitigate_adversarial_no_model_raises(self, mitigator):
        X = np.random.default_rng(42).normal(size=(10, 2))
        y = np.random.default_rng(42).integers(0, 2, size=10)
        sensitive = np.random.default_rng(42).integers(0, 2, size=10)
        with pytest.raises(ValueError, match="model"):
            mitigator.mitigate(X, y, sensitive, strategy="adversarial")

    def test_mitigate_postprocessing(self, mitigator):
        X = np.random.default_rng(42).normal(size=(50, 3))
        y = np.random.default_rng(42).integers(0, 2, size=50)
        sensitive = np.random.default_rng(42).integers(0, 2, size=50)
        y_pred = np.random.default_rng(42).uniform(0, 1, size=50)
        result = mitigator.mitigate(
            X, y, sensitive, strategy="postprocessing", base_predictions=y_pred
        )
        assert result.strategy_used == "postprocessing"

    def test_mitigate_reject_option(self, mitigator):
        X = np.random.default_rng(42).normal(size=(50, 3))
        y = np.random.default_rng(42).integers(0, 2, size=50)
        sensitive = np.random.default_rng(42).integers(0, 2, size=50)
        y_pred = np.random.default_rng(42).uniform(0, 1, size=50)
        result = mitigator.mitigate(
            X, y, sensitive, strategy="reject_option", base_predictions=y_pred
        )
        assert result.strategy_used == "reject_option"

    def test_mitigate_invalid_strategy(self, mitigator):
        X = np.random.default_rng(42).normal(size=(10, 2))
        y = np.random.default_rng(42).integers(0, 2, size=10)
        sensitive = np.random.default_rng(42).integers(0, 2, size=10)
        with pytest.raises(ValueError, match="Unknown strategy"):
            mitigator.mitigate(X, y, sensitive, strategy="invalid_strategy")

    def test_evaluate_mitigation(self, mitigator):
        X = np.random.default_rng(42).normal(size=(50, 3))
        y = np.random.default_rng(42).integers(0, 2, size=50)
        y_pred = np.random.default_rng(42).uniform(0, 1, size=50)
        sensitive = np.random.default_rng(42).integers(0, 2, size=50)
        result = mitigator.evaluate_mitigation(X, y, y_pred, sensitive, strategy="reweighing")
        assert isinstance(result, MitigationResult)
        assert "before_metrics" in result.__dict__
        assert "after_metrics" in result.__dict__

    def test_mitigation_result_dataclass(self):
        result = MitigationResult(strategy_used="reweighing")
        assert result.strategy_used == "reweighing"
        assert result.before_metrics == {}
        assert result.improvement == {}

    def test_reweighing_returns_correct_shape(self, mitigator):
        X = np.random.default_rng(42).normal(size=(20, 2))
        y = np.array([1, 0] * 10)
        sensitive = np.array([0] * 10 + [1] * 10)
        weights = mitigator.reweighing(X, y, sensitive)
        assert weights.shape == (20,)

    def test_equalized_odds_small_group_fallback(self, mitigator):
        """Test that groups with <5 samples are skipped."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        sensitive = np.array([0, 0, 0, 0, 0, 1])
        adjusted = mitigator.equalized_odds_postprocessing(y_pred, sensitive, y_true)
        assert adjusted.shape == y_pred.shape
