"""Tests for the calibration and uncertainty toolkit and its API router."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.calibration import router as calibration_router
from astroml.training.calibration.bayesian import BayesianUncertainty
from astroml.training.calibration.conformal import ConformalPredictor
from astroml.training.calibration.isotonic import IsotonicCalibrator
from astroml.training.calibration.platt import PlattCalibrator

app = FastAPI()
app.include_router(calibration_router)
client = TestClient(app)


@pytest.fixture
def calibration_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a miscalibrated probability dataset."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.clip(0.5 + 0.4 * (y_true - 0.5) + rng.normal(0, 0.2, 200), 0.01, 0.99)
    return y_true, y_prob


class TestPlattCalibrator:
    def test_fit_and_calibrate(self, calibration_data):
        y_true, y_prob = calibration_data
        calibrator = PlattCalibrator().fit(y_prob, y_true)
        assert calibrator.fitted
        assert calibrator.a != 0.0
        calibrated = calibrator.calibrate(y_prob)
        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_calibrate_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fitted"):
            PlattCalibrator().calibrate(np.array([0.5]))

    def test_calibrate_scores(self, calibration_data):
        y_true, y_prob = calibration_data
        calibrator = PlattCalibrator().fit(y_prob, y_true)
        scores = np.array([-1.0, 0.0, 1.0])
        calibrated = calibrator.calibrate_scores(scores)
        assert np.all(calibrated > 0) and np.all(calibrated < 1)

    def test_invalid_probabilities_raise(self, calibration_data):
        y_true, y_prob = calibration_data
        with pytest.raises(ValueError, match="between 0 and 1"):
            PlattCalibrator().fit(np.array([1.5, 0.5]), np.array([1, 0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            PlattCalibrator().fit(np.array([0.5, 0.6]), np.array([1]))

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="At least two"):
            PlattCalibrator().fit(np.array([0.5]), np.array([1]))

    def test_non_binary_labels_raise(self):
        with pytest.raises(ValueError, match="binary"):
            PlattCalibrator().fit(np.array([0.5, 0.6]), np.array([1, 2]))

    def test_calibrate_invalid_input(self, calibration_data):
        y_true, y_prob = calibration_data
        calibrator = PlattCalibrator().fit(y_prob, y_true)
        with pytest.raises(ValueError, match="between 0 and 1"):
            calibrator.calibrate(np.array([2.0]))


class TestIsotonicCalibrator:
    def test_fit_and_calibrate(self, calibration_data):
        y_true, y_prob = calibration_data
        calibrator = IsotonicCalibrator().fit(y_prob, y_true)
        assert calibrator.fitted
        calibrated = calibrator.calibrate(y_prob)
        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_monotonic_predictions(self, calibration_data):
        y_true, y_prob = calibration_data
        calibrator = IsotonicCalibrator().fit(y_prob, y_true)
        calibrated = calibrator.calibrate(y_prob)
        order = np.argsort(y_prob)
        assert np.all(np.diff(calibrated[order]) >= -1e-9)

    def test_calibrate_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fitted"):
            IsotonicCalibrator().calibrate(np.array([0.5]))

    def test_invalid_probabilities_raise(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            IsotonicCalibrator().fit(np.array([-0.1, 0.5]), np.array([1, 0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            IsotonicCalibrator().fit(np.array([0.5]), np.array([1, 0]))


class TestConformalPredictor:
    def test_classification_sets(self):
        probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
        y_cal = np.array([0, 1, 1, 0])
        predictor = ConformalPredictor().fit_classification(probs, y_cal)
        result = predictor.predict_sets(probs, significance=0.1, y_true=y_cal)
        assert result.prediction_sets
        assert all(isinstance(s, list) for s in result.prediction_sets)
        assert result.coverage is not None
        assert result.quantile > 0

    def test_coverage_without_labels(self):
        probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        predictor = ConformalPredictor().fit_classification(probs, np.array([0, 1]))
        result = predictor.predict_sets(probs, significance=0.2)
        assert result.coverage is None

    def test_regression_intervals(self):
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.1, 1.9, 3.2])
        predictor = ConformalPredictor().fit_regression(y_pred, y_true)
        result = predictor.predict_intervals(y_pred, significance=0.1, y_true=y_true)
        assert len(result.lower_bounds) == 3
        assert result.lower_bounds[0] < result.upper_bounds[0]
        assert result.coverage == 1.0

    def test_regression_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_regression"):
            ConformalPredictor().predict_intervals(np.array([1.0]))

    def test_classification_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_classification"):
            ConformalPredictor().predict_sets(np.array([[0.5, 0.5]]))

    def test_invalid_significance(self):
        probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        predictor = ConformalPredictor().fit_classification(probs, np.array([0, 1]))
        with pytest.raises(ValueError, match="significance"):
            predictor.predict_sets(probs, significance=1.5)

    def test_invalid_probs_raise(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConformalPredictor().fit_classification(np.array([[1.5, 0.5]]), np.array([0]))

    def test_probs_not_summing_to_one_raise(self):
        with pytest.raises(ValueError, match="sum to 1"):
            ConformalPredictor().fit_classification(np.array([[0.5, 0.2]]), np.array([0]))

    def test_regression_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            ConformalPredictor().fit_regression(np.array([1.0, 2.0]), np.array([1.0]))

    def test_y_true_mismatch_raises(self):
        probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        predictor = ConformalPredictor().fit_classification(probs, np.array([0, 1]))
        with pytest.raises(ValueError, match="match"):
            predictor.predict_sets(probs, y_true=np.array([0, 1, 0]))


class TestBayesianUncertainty:
    def test_predictive_entropy(self):
        probs = np.array([[0.9, 0.1], [0.5, 0.5]])
        entropy = BayesianUncertainty().predictive_entropy(probs)
        assert entropy[0] < entropy[1]
        assert entropy[1] > 0

    def test_predictive_variance(self):
        probs = np.array([[0.9, 0.1], [0.5, 0.5]])
        variance = BayesianUncertainty().predictive_variance(probs)
        assert variance[0] < variance[1]
        assert np.all(variance >= 0)

    def test_mutual_information(self):
        samples = np.array(
            [
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.1, 0.9], [0.9, 0.1]],
            ]
        )
        mi = BayesianUncertainty().mutual_information(samples)
        assert mi.shape == (2,)
        assert mi[0] > 0  # disagreement -> high MI
        assert mi[1] == pytest.approx(0.0, abs=1e-9)  # agreement -> zero MI

    def test_mutual_information_wrong_ndim(self):
        with pytest.raises(ValueError, match="3D"):
            BayesianUncertainty().mutual_information(np.array([[0.5, 0.5]]))

    def test_monte_carlo_dropout(self):
        model = object()

        def predict_fn(_model, X):
            return np.full((len(X), 2), 0.5)

        X = np.array([[1.0], [2.0]])
        mean, std = BayesianUncertainty().monte_carlo_dropout(model, X, predict_fn, n_iterations=5)
        assert mean.shape == (2, 2)
        assert np.allclose(mean, 0.5)
        assert np.allclose(std, 0.0)

    def test_monte_carlo_dropout_invalid_iterations(self):
        with pytest.raises(ValueError, match="positive"):
            BayesianUncertainty().monte_carlo_dropout(
                object(), np.array([[1.0]]), lambda m, x: x, n_iterations=0
            )

    def test_bootstrap_uncertainty(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)

        def fit_predict(X_train, y_train, X_test):
            return np.full(len(X_test), 0.5)

        mean, std = BayesianUncertainty().bootstrap_uncertainty(
            X, y, fit_predict, n_bootstraps=5, sample_fraction=0.8
        )
        assert mean.shape == (50,)
        assert std.shape == (50,)
        assert np.allclose(mean, 0.5)

    def test_bootstrap_invalid_bootstraps(self):
        with pytest.raises(ValueError, match="positive"):
            BayesianUncertainty().bootstrap_uncertainty(
                np.array([[1.0]]), np.array([0]), lambda a, b, c: c, n_bootstraps=0
            )

    def test_bootstrap_invalid_fraction(self):
        with pytest.raises(ValueError, match="sample_fraction"):
            BayesianUncertainty().bootstrap_uncertainty(
                np.array([[1.0]]), np.array([0]), lambda a, b, c: c, sample_fraction=1.5
            )

    def test_invalid_probs_raise(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BayesianUncertainty().predictive_entropy(np.array([[1.5, 0.5]]))


class TestCalibrationAPI:
    def test_platt_success(self):
        response = client.post(
            "/api/v1/calibration/platt",
            json={"y_true": [1, 0, 1, 0, 1], "y_prob": [0.9, 0.1, 0.8, 0.2, 0.7]},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "a" in data and "b" in data
        assert "brier_before" in data and "brier_after" in data
        assert len(data["calibrated_probabilities"]) == 5

    def test_platt_invalid(self):
        response = client.post(
            "/api/v1/calibration/platt",
            json={"y_true": [1, 0], "y_prob": [1.5, 0.2]},
        )
        assert response.status_code == 400

    def test_isotonic_success(self):
        response = client.post(
            "/api/v1/calibration/isotonic",
            json={"y_true": [1, 0, 1, 0, 1], "y_prob": [0.9, 0.1, 0.8, 0.2, 0.7]},
        )
        assert response.status_code == 200
        assert len(response.json()["data"]["calibrated_probabilities"]) == 5

    def test_conformal_success(self):
        payload = {
            "probs_cal": [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]],
            "y_cal": [0, 1, 1, 0],
            "probs": [[0.9, 0.1], [0.2, 0.8]],
            "y_true": [0, 1],
            "significance": 0.1,
        }
        response = client.post("/api/v1/calibration/conformal", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["prediction_sets"]) == 2
        assert data["coverage"] is not None

    def test_conformal_bad_significance(self):
        payload = {
            "probs_cal": [[0.9, 0.1], [0.6, 0.4]],
            "y_cal": [0, 1],
            "probs": [[0.9, 0.1]],
            "significance": 1.5,
        }
        response = client.post("/api/v1/calibration/conformal", json=payload)
        assert response.status_code == 400

    def test_uncertainty_success(self):
        response = client.post(
            "/api/v1/calibration/uncertainty",
            json={"probs": [[0.9, 0.1], [0.5, 0.5]]},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["entropy"]) == 2
        assert data["entropy"][0] < data["entropy"][1]

    def test_uncertainty_invalid(self):
        response = client.post(
            "/api/v1/calibration/uncertainty",
            json={"probs": [[1.5, 0.1]]},
        )
        assert response.status_code == 400
