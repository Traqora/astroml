"""Tests for the model debugging toolkit and its API router."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.debugging import router as debugging_router
from astroml.training.debugging.confusion_analysis import ConfusionAnalyzer
from astroml.training.debugging.error_analysis import ErrorAnalyzer
from astroml.training.debugging.failure_modes import FailureModeIdentifier
from astroml.training.debugging.slice_analysis import SliceAnalyzer

app = FastAPI()
app.include_router(debugging_router)
client = TestClient(app)


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a small dataset with known errors."""
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    return y_true, y_pred


class TestErrorAnalyzer:
    def test_error_indices(self, sample_data):
        y_true, y_pred = sample_data
        indices = ErrorAnalyzer().error_indices(y_true, y_pred)
        assert indices.tolist() == [2, 3, 5, 7]

    def test_analyze_basic(self, sample_data):
        y_true, y_pred = sample_data
        result = ErrorAnalyzer().analyze(y_true, y_pred)
        assert result.total_samples == 8
        assert result.error_count == 4
        assert result.error_rate == 0.5
        assert result.accuracy == 0.5
        assert result.error_indices.tolist() == [2, 3, 5, 7]

    def test_analyze_perfect_predictions(self):
        y = np.array([0, 1, 1, 0])
        result = ErrorAnalyzer().analyze(y, y.copy())
        assert result.error_count == 0
        assert result.accuracy == 1.0
        assert result.class_metrics[0].error_rate == 0.0

    def test_class_metrics(self, sample_data):
        y_true, y_pred = sample_data
        metrics = ErrorAnalyzer().error_rate_by_class(y_true, y_pred)
        assert set(metrics.keys()) == {0, 1}
        assert metrics[0].total == 4
        assert metrics[0].incorrect == 2
        assert metrics[1].incorrect == 2

    def test_error_distribution(self, sample_data):
        y_true, y_pred = sample_data
        distribution = ErrorAnalyzer().error_distribution(y_true, y_pred)
        assert distribution[(1, 0)] == 2
        assert distribution[(0, 1)] == 2

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            ErrorAnalyzer().analyze(np.array([1, 0]), np.array([1]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ErrorAnalyzer().analyze(np.array([]), np.array([]))


class TestConfusionAnalyzer:
    def test_compute_basic(self, sample_data):
        y_true, y_pred = sample_data
        result = ConfusionAnalyzer().compute(y_true, y_pred)
        assert result.labels == [0, 1]
        assert result.matrix.shape == (2, 2)
        assert result.matrix[0, 0] == 2  # true 0, pred 0
        assert result.matrix[1, 1] == 2  # true 1, pred 1
        assert result.accuracy == 0.5

    def test_class_metrics_values(self, sample_data):
        y_true, y_pred = sample_data
        result = ConfusionAnalyzer().compute(y_true, y_pred)
        metrics = result.class_metrics[1]
        assert metrics.tp == 2
        assert metrics.fp == 2
        assert metrics.fn == 2
        assert metrics.tn == 2
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5
        assert metrics.f1 == 0.5
        assert metrics.support == 4

    def test_normalize_modes(self):
        matrix = np.array([[2, 0], [0, 2]])
        analyzer = ConfusionAnalyzer()
        assert np.allclose(analyzer.normalize(matrix, "true"), np.eye(2))
        assert np.allclose(analyzer.normalize(matrix, "pred"), np.eye(2))
        assert np.allclose(analyzer.normalize(matrix, "all"), np.array([[0.5, 0.0], [0.0, 0.5]]))
        assert np.array_equal(analyzer.normalize(matrix, "none"), matrix)

    def test_normalize_zero_rows(self):
        matrix = np.array([[0, 0], [1, 1]])
        normalized = ConfusionAnalyzer().normalize(matrix, "true")
        assert np.allclose(normalized[0], [0, 0])
        assert np.allclose(normalized[1], [0.5, 0.5])

    def test_normalize_invalid_mode(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ConfusionAnalyzer().normalize(np.eye(2), "bogus")

    def test_custom_labels(self):
        y_true = np.array([2, 3, 2, 3])
        y_pred = np.array([2, 2, 3, 3])
        result = ConfusionAnalyzer().compute(y_true, y_pred, labels=[2, 3])
        assert result.labels == [2, 3]
        assert result.matrix.shape == (2, 2)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            ConfusionAnalyzer().compute(np.array([1, 0]), np.array([1]))

    def test_to_dict(self, sample_data):
        y_true, y_pred = sample_data
        result = ConfusionAnalyzer().compute(y_true, y_pred)
        data = ConfusionAnalyzer().to_dict(result)
        assert data["labels"] == [0, 1]
        assert data["matrix"] == result.matrix.tolist()
        assert "class_metrics" in data
        assert data["class_metrics"]["1"]["f1"] == 0.5


class TestSliceAnalyzer:
    def test_analyze_slices(self, sample_data):
        y_true, y_pred = sample_data
        slices = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        result = SliceAnalyzer().analyze(y_true, y_pred, slices)
        assert len(result.slices) == 2
        slice_a = next(s for s in result.slices if s.slice_name == "a")
        slice_b = next(s for s in result.slices if s.slice_name == "b")
        assert slice_a.support == 4
        assert slice_a.accuracy == 0.5
        assert slice_b.accuracy == 0.5

    def test_worst_slice(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        slices = np.array(["good", "good", "bad", "bad"])
        result = SliceAnalyzer().analyze(y_true, y_pred, slices)
        assert result.worst_slice == "bad"
        assert result.worst_accuracy == 0.0
        assert result.overall_accuracy == 0.5

    def test_underperforming_slices(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        slices = np.array(["good", "good", "bad", "bad"])
        under = SliceAnalyzer().underperforming_slices(y_true, y_pred, slices, threshold=0.8)
        assert [s.slice_name for s in under] == ["bad"]

    def test_metrics_fields(self, sample_data):
        y_true, y_pred = sample_data
        slices = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        result = SliceAnalyzer().analyze(y_true, y_pred, slices)
        metrics = result.slices[0]
        assert metrics.correct + metrics.incorrect == metrics.support
        assert metrics.error_rate == 1 - metrics.accuracy
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            SliceAnalyzer().analyze(np.array([1, 0]), np.array([1, 0]), np.array(["a"]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SliceAnalyzer().analyze(np.array([]), np.array([]), np.array([]))


class TestFailureModeIdentifier:
    def test_identify_basic(self, sample_data):
        y_true, y_pred = sample_data
        report = FailureModeIdentifier().identify(y_true, y_pred)
        names = [mode.name for mode in report.modes]
        assert names == ["false_positives", "false_negatives"]
        assert report.total_errors == 4

    def test_false_positive_mode(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 0])
        report = FailureModeIdentifier().identify(y_true, y_pred)
        fp = report.modes[0]
        assert fp.name == "false_positives"
        assert fp.count == 1
        assert fp.sample_indices == [0]

    def test_high_confidence_errors(self, sample_data):
        y_true, y_pred = sample_data
        y_prob = np.array([0.9, 0.6, 0.95, 0.85, 0.7, 0.9, 0.5, 0.8])
        report = FailureModeIdentifier().identify(y_true, y_pred, y_prob=y_prob)
        names = [mode.name for mode in report.modes]
        assert "high_confidence_errors" in names
        hc = next(m for m in report.modes if m.name == "high_confidence_errors")
        assert hc.count >= 1

    def test_slice_failures(self, sample_data):
        y_true, y_pred = sample_data
        slices = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        report = FailureModeIdentifier().identify(y_true, y_pred, slice_labels=slices)
        names = [mode.name for mode in report.modes]
        assert "slice_failures" in names

    def test_severity_levels(self):
        y_true = np.zeros(10, dtype=int)
        y_pred = np.ones(10, dtype=int)
        report = FailureModeIdentifier().identify(y_true, y_pred)
        fp = report.modes[0]
        assert fp.severity == "high"
        assert fp.rate == 1.0

    def test_summarize(self, sample_data):
        y_true, y_pred = sample_data
        report = FailureModeIdentifier().identify(y_true, y_pred)
        summary = FailureModeIdentifier().summarize(report)
        assert summary["total_errors"] == 4
        assert len(summary["modes"]) == 2

    def test_confidence_threshold(self, sample_data):
        y_true, y_pred = sample_data
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        report = FailureModeIdentifier(confidence_threshold=0.9).identify(
            y_true, y_pred, y_prob=y_prob
        )
        hc = next(m for m in report.modes if m.name == "high_confidence_errors")
        assert hc.count == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            FailureModeIdentifier().identify(np.array([1, 0]), np.array([1]))

    def test_prob_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            FailureModeIdentifier().identify(
                np.array([1, 0]), np.array([1, 0]), y_prob=np.array([0.5])
            )

    def test_slice_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            FailureModeIdentifier().identify(
                np.array([1, 0]),
                np.array([1, 0]),
                slice_labels=np.array(["a"]),
            )


class TestDebuggingAPI:
    def test_error_analysis_success(self):
        response = client.post(
            "/api/v1/debugging/error-analysis",
            json={"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 0, 1]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["error_count"] == 2

    def test_error_analysis_bad_shape(self):
        response = client.post(
            "/api/v1/debugging/error-analysis",
            json={"y_true": [1, 0], "y_pred": [1]},
        )
        assert response.status_code == 400

    def test_confusion_matrix_success(self):
        response = client.post(
            "/api/v1/debugging/confusion-matrix",
            json={"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 1, 0], "norm": "true"},
        )
        assert response.status_code == 200
        assert response.json()["data"]["accuracy"] == 1.0

    def test_confusion_matrix_invalid_norm(self):
        response = client.post(
            "/api/v1/debugging/confusion-matrix",
            json={"y_true": [1, 0], "y_pred": [1, 0], "norm": "bogus"},
        )
        assert response.status_code == 400

    def test_slices_success(self):
        response = client.post(
            "/api/v1/debugging/slices",
            json={
                "y_true": [1, 1, 1, 1],
                "y_pred": [1, 1, 0, 0],
                "slice_labels": ["good", "good", "bad", "bad"],
                "threshold": 0.8,
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["worst_slice"] == "bad"
        assert data["underperforming_slices"] == ["bad"]

    def test_failure_modes_success(self):
        response = client.post(
            "/api/v1/debugging/failure-modes",
            json={"y_true": [1, 0, 1], "y_pred": [1, 0, 0], "y_prob": [0.9, 0.5, 0.9]},
        )
        assert response.status_code == 200
        modes = response.json()["data"]["modes"]
        assert any(mode["name"] == "high_confidence_errors" for mode in modes)

    def test_failure_modes_mismatch(self):
        response = client.post(
            "/api/v1/debugging/failure-modes",
            json={"y_true": [1, 0], "y_pred": [1]},
        )
        assert response.status_code == 400

    def test_extra_fields_forbidden(self):
        response = client.post(
            "/api/v1/debugging/error-analysis",
            json={"y_true": [1, 0], "y_pred": [1, 0], "extra": 1},
        )
        assert response.status_code == 422
