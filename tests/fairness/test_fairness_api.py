"""Tests for fairness API router."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.fairness import router as fairness_router

app = FastAPI()
app.include_router(fairness_router)

client = TestClient(app)


class TestFairnessAPI:
    def test_compute_metrics_success(self):
        payload = {
            "y_true": [1, 0, 1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0, 1, 0],
            "sensitive_features": [0, 0, 1, 1, 0, 1],
        }
        response = client.post("/api/v1/fairness/metrics", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "demographic_parity" in data["data"]

    def test_compute_metrics_empty_data(self):
        payload = {
            "y_true": [],
            "y_pred": [],
            "sensitive_features": [],
        }
        response = client.post("/api/v1/fairness/metrics", json=payload)
        assert response.status_code == 400

    def test_compute_metrics_shape_mismatch(self):
        payload = {
            "y_true": [1, 0],
            "y_pred": [1],
            "sensitive_features": [0, 1],
        }
        response = client.post("/api/v1/fairness/metrics", json=payload)
        assert response.status_code == 400

    def test_detect_bias_success(self):
        payload = {
            "y_true": [1, 0, 1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0, 1, 0],
            "sensitive_features": [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]],
            "attributes": ["gender", "race"],
        }
        response = client.post("/api/v1/fairness/bias/detect", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "overall_bias_detected" in data["data"]

    def test_detect_bias_one_d(self):
        payload = {
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0],
            "sensitive_features": [0, 0, 1, 1],
        }
        response = client.post("/api/v1/fairness/bias/detect", json=payload)
        assert response.status_code == 200

    def test_intersectional_analysis_success(self):
        payload = {
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0],
            "sensitive_features": [[0, 1], [0, 1], [1, 0], [1, 0]],
        }
        response = client.post("/api/v1/fairness/bias/intersectional", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_mitigate_success(self):
        payload = {
            "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            "y": [1, 0, 1, 0],
            "sensitive_features": [0, 0, 1, 1],
            "strategy": "reweighing",
        }
        response = client.post("/api/v1/fairness/mitigate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["strategy_used"] == "reweighing"

    def test_report_success(self):
        payload = {
            "y_true": [1, 0, 1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0, 1, 0],
            "sensitive_features": [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]],
            "attributes": ["gender", "race"],
        }
        response = client.post("/api/v1/fairness/report", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "overall_scores" in data["data"]

    def test_mitigate_invalid_strategy(self):
        payload = {
            "X": [[1.0, 2.0], [3.0, 4.0]],
            "y": [1, 0],
            "sensitive_features": [0, 1],
            "strategy": "nonexistent",
        }
        response = client.post("/api/v1/fairness/mitigate", json=payload)
        assert response.status_code == 400 or response.status_code == 500
