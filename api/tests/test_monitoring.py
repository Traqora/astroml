"""
Integration tests — model monitoring (issue #244).

Covers: metrics endpoint shape, history endpoint, latency recording,
drift report structure, and prediction stats.
"""
from __future__ import annotations

import pytest


@pytest.mark.xdist_group("api_monitoring")
class TestMonitoringMetrics:
    """Verify /api/v1/monitoring/metrics returns required fields."""

    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/monitoring/metrics")
        assert resp.status_code == 200

    def test_metrics_response_has_required_fields(self, client):
        data = client.get("/api/v1/monitoring/metrics").json()
        required = {"accuracy", "precision", "recall", "f1", "f1_score", "auc", "auc_roc", "drift_score"}
        assert required.issubset(data.keys()), f"missing keys: {required - data.keys()}"

    def test_metrics_values_are_numeric_or_null(self, client):
        data = client.get("/api/v1/monitoring/metrics").json()
        for key in ("accuracy", "precision", "recall", "f1", "f1_score", "auc", "auc_roc"):
            assert data[key] is None or isinstance(data[key], (int, float))


@pytest.mark.xdist_group("api_monitoring")
class TestMonitoringHistory:
    """Verify /api/v1/monitoring/performance-history returns a list."""

    def test_history_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/monitoring/performance-history")
        assert resp.status_code == 200

    def test_history_is_list(self, client):
        data = client.get("/api/v1/monitoring/performance-history").json()
        assert isinstance(data, list)


@pytest.mark.xdist_group("api_monitoring")
class TestMonitoringDriftReport:
    """Verify /api/v1/monitoring/drift-report returns expected structure."""

    def test_drift_report_returns_200(self, client):
        resp = client.get("/api/v1/monitoring/drift-report")
        assert resp.status_code == 200

    def test_drift_report_has_features(self, client):
        data = client.get("/api/v1/monitoring/drift-report").json()
        assert "features" in data


@pytest.mark.xdist_group("api_monitoring")
class TestMonitoringLatency:
    """Verify /api/v1/monitoring/latency returns expected structure."""

    def test_latency_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/monitoring/latency")
        assert resp.status_code == 200

    def test_latency_has_percentile_fields(self, client):
        data = client.get("/api/v1/monitoring/latency").json()
        assert "p50_ms" in data
        assert "p95_ms" in data
        assert "p99_ms" in data

    def test_latency_values_non_negative(self, client):
        data = client.get("/api/v1/monitoring/latency").json()
        for key in ("p50_ms", "p95_ms", "p99_ms"):
            assert data[key] >= 0


@pytest.mark.xdist_group("api_monitoring")
class TestMonitoringPredictionStats:
    """Verify /api/v1/monitoring/prediction-stats structure."""

    def test_prediction_stats_returns_200(self, client):
        resp = client.get("/api/v1/monitoring/prediction-stats")
        assert resp.status_code == 200

    def test_prediction_stats_has_total(self, client):
        data = client.get("/api/v1/monitoring/prediction-stats").json()
        assert "total_predictions" in data
        assert isinstance(data["total_predictions"], int)
