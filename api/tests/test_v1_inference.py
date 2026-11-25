from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from astroml.api.routers import v1_inference
from astroml.api.v1_app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness_are_available_without_external_services(client):
    health = client.get("/api/v1/health")
    readiness = client.get("/api/v1/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"
    assert readiness.json()["required_dependencies"] == {
        "numpy": True,
        "registered_predictor": True,
    }


def test_readiness_fails_when_no_predictor_is_registered(client, monkeypatch):
    monkeypatch.setattr(v1_inference, "_PREDICTORS", {})

    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["required_dependencies"]["registered_predictor"] is False


def test_prediction_response_has_stable_v1_shape(client):
    response = client.post(
        "/api/v1/predict",
        json={"features": [[1.0, 3.0], [2.0, 4.0]]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "v1",
        "model_name": "mean_baseline",
        "count": 2,
        "scores": [2.0, 3.0],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [[1.0], [2.0, 3.0]]},
        {"features": [[1.0]], "unexpected": True},
        {"features": [[float("nan")]]},
        {"features": [[1.0] * (v1_inference.MAX_FEATURES_PER_ROW + 1)]},
        {"features": [[1.0]] * (v1_inference.MAX_PREDICTION_ROWS + 1)},
    ],
)
def test_prediction_rejects_malformed_or_oversized_requests(client, payload):
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_unknown_model(client):
    response = client.post(
        "/api/v1/predict",
        json={"model_name": "missing", "features": [[1.0]]},
    )

    assert response.status_code == 404


def test_prediction_rejects_invalid_predictor_output(client, monkeypatch):
    monkeypatch.setitem(v1_inference._PREDICTORS, "bad", lambda features: [1.0, 2.0])

    response = client.post(
        "/api/v1/predict",
        json={"model_name": "bad", "features": [[1.0]]},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "predictor must return exactly one score per input row"