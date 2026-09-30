"""Data quality tests for calibration score outputs.

Focuses on input validation, metric range correctness, and edge-case
robustness of CalibrationAnalyzer.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from astroml.validation.calibration import CalibrationAnalyzer, create_sample_fraud_data


class TestInputValidation:
    """Input data quality gates enforced by CalibrationAnalyzer."""

    def test_rejects_probabilities_above_one(self):
        analyzer = CalibrationAnalyzer()
        with pytest.raises(ValueError, match="y_prob must be between 0 and 1"):
            analyzer.compute_calibration_curve(np.array([0, 1, 0]), np.array([0.2, 1.5, 0.3]))

    def test_rejects_negative_probabilities(self):
        analyzer = CalibrationAnalyzer()
        with pytest.raises(ValueError, match="y_prob must be between 0 and 1"):
            analyzer.compute_calibration_curve(np.array([0, 1, 0]), np.array([-0.1, 0.8, 0.3]))

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            CalibrationAnalyzer().compute_calibration_curve(
                np.array([0, 1]), np.array([0.2, 0.8, 0.5])
            )

    def test_accepts_clipped_boundary_values(self):
        """Probabilities at 0.01 / 0.99 are valid inputs."""
        analyzer = CalibrationAnalyzer(n_bins=4)
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.01, 0.99, 0.01, 0.99])
        fraction_pos, mean_pred = analyzer.compute_calibration_curve(y_true, y_prob)
        assert len(fraction_pos) >= 1


class TestMetricBounds:
    """All calibration metrics must lie within their defined value domains."""

    def test_all_expected_metrics_present(self, fraud_scores):
        y_true, y_prob = fraud_scores
        metrics = CalibrationAnalyzer().compute_calibration_metrics(y_true, y_prob)
        for key in (
            "brier_score",
            "log_loss",
            "ece",
            "mce",
            "ace",
            "overconfidence",
            "underconfidence",
            "sharpness",
        ):
            assert key in metrics

    def test_brier_score_in_unit_interval(self, fraud_scores):
        y_true, y_prob = fraud_scores
        m = CalibrationAnalyzer().compute_calibration_metrics(y_true, y_prob)
        assert 0.0 <= m["brier_score"] <= 1.0

    def test_log_loss_nonnegative(self, fraud_scores):
        y_true, y_prob = fraud_scores
        m = CalibrationAnalyzer().compute_calibration_metrics(y_true, y_prob)
        assert m["log_loss"] >= 0.0

    def test_calibration_errors_in_unit_interval(self, fraud_scores):
        y_true, y_prob = fraud_scores
        m = CalibrationAnalyzer().compute_calibration_metrics(y_true, y_prob)
        for key in ("ece", "mce", "ace"):
            assert 0.0 <= m[key] <= 1.0, f"{key} out of [0, 1]: {m[key]}"

    def test_sharpness_nonnegative(self, fraud_scores):
        y_true, y_prob = fraud_scores
        m = CalibrationAnalyzer().compute_calibration_metrics(y_true, y_prob)
        assert m["sharpness"] >= 0.0

    def test_no_metric_is_nan(self, fraud_scores):
        y_true, y_prob = fraud_scores
        m = CalibrationAnalyzer().compute_calibration_metrics(y_true, y_prob)
        for name, value in m.items():
            assert not np.isnan(value), f"{name} returned NaN"


class TestEdgeCaseDistributions:
    """Calibration metrics remain finite on degenerate score distributions."""

    def test_all_same_prediction(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        m = CalibrationAnalyzer(n_bins=4).compute_calibration_metrics(y_true, y_prob)
        assert all(not np.isnan(v) for v in m.values())

    def test_all_legitimate_class(self):
        y_true = np.zeros(20, dtype=int)
        y_prob = np.linspace(0.01, 0.4, 20)
        m = CalibrationAnalyzer(n_bins=5).compute_calibration_metrics(y_true, y_prob)
        assert all(not np.isnan(v) for v in m.values())

    def test_all_fraud_class(self):
        y_true = np.ones(20, dtype=int)
        y_prob = np.linspace(0.6, 0.99, 20)
        m = CalibrationAnalyzer(n_bins=5).compute_calibration_metrics(y_true, y_prob)
        assert all(not np.isnan(v) for v in m.values())

    def test_quantile_strategy(self, fraud_scores):
        y_true, y_prob = fraud_scores
        analyzer = CalibrationAnalyzer(n_bins=10, strategy="quantile")
        fraction_pos, mean_pred = analyzer.compute_calibration_curve(y_true, y_prob)
        assert len(fraction_pos) > 0
        assert all(0 <= fp <= 1 for fp in fraction_pos)


class TestSampleDataQuality:
    """Generated sample data satisfies basic quality invariants."""

    def test_shapes_match(self):
        y_true, y_prob = create_sample_fraud_data(n_samples=200)
        assert y_true.shape == y_prob.shape == (200,)

    def test_labels_are_binary(self):
        y_true, _ = create_sample_fraud_data(n_samples=200)
        assert set(np.unique(y_true)) <= {0, 1}

    def test_probabilities_in_valid_range(self):
        _, y_prob = create_sample_fraud_data(n_samples=200)
        assert np.all(y_prob >= 0) and np.all(y_prob <= 1)

    def test_fraud_rate_near_target(self):
        y_true, _ = create_sample_fraud_data(n_samples=2000, fraud_rate=0.15)
        assert abs(np.mean(y_true) - 0.15) < 0.05

    def test_reproducible_with_fixed_seed(self):
        y1, p1 = create_sample_fraud_data(n_samples=100)
        y2, p2 = create_sample_fraud_data(n_samples=100)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(p1, p2)


class TestReportGeneration:
    """Calibration report must contain required quality indicators."""

    def test_model_name_in_report(self, fraud_scores):
        y_true, y_prob = fraud_scores
        report = CalibrationAnalyzer().generate_calibration_report(
            y_true, y_prob, model_name="QA_Validator"
        )
        assert "QA_Validator" in report

    def test_required_sections_present(self, fraud_scores):
        y_true, y_prob = fraud_scores
        report = CalibrationAnalyzer().generate_calibration_report(y_true, y_prob)
        for section in ("Brier Score", "Expected Calibration Error", "Recommendations"):
            assert section in report, f"Missing section: {section}"

    def test_plot_produces_four_subplots(self, fraud_scores):
        y_true, y_prob = fraud_scores
        with patch("matplotlib.pyplot.show"):
            fig = CalibrationAnalyzer(n_bins=8).plot_calibration_curve(
                y_true, y_prob, "QA_Plot_Test"
            )
        assert len(fig.axes) == 4
