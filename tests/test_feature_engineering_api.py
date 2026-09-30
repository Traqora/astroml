"""Tests for Automated Feature Engineering API Router."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.feature_engineering import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "transactions": [
            {
                "transaction_id": f"tx_{i}",
                "source_account": f"acc_{i % 3}",
                "destination_account": f"acc_{(i + 1) % 3}",
                "amount": float(10 * (i + 1)),
                "fee": 0.01,
                "timestamp": f"2026-01-01T{i:02d}:00:00",
            }
            for i in range(10)
        ],
        "accounts": [
            {"account_id": "acc_0", "account_type": "retail"},
            {"account_id": "acc_1", "account_type": "merchant"},
            {"account_id": "acc_2", "account_type": "retail"},
        ],
        "max_depth": 2,
        "target_entity": "accounts",
    }


def test_health_check(client):
    response = client.get("/api/v1/feature-engineering/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_primitives_endpoint(client):
    response = client.get("/api/v1/feature-engineering/primitives")
    assert response.status_code == 200
    data = response.json()
    assert "aggregation_primitives" in data
    assert "transform_primitives" in data
    assert len(data["aggregation_primitives"]) > 0


def test_synthesize_endpoint(client, sample_payload):
    response = client.post("/api/v1/feature-engineering/synthesize", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_features"] > 0
    assert len(data["feature_names"]) > 0
    assert len(data["sample_records"]) > 0


def test_select_features_endpoint(client):
    payload = {
        "feature_data": [
            {"feat1": 1.0, "feat2": 5.0, "const_feat": 10.0},
            {"feat1": 2.0, "feat2": 15.0, "const_feat": 10.0},
            {"feat1": 3.0, "feat2": 25.0, "const_feat": 10.0},
            {"feat1": 4.0, "feat2": 35.0, "const_feat": 10.0},
        ],
        "variance_threshold": 0.01,
        "correlation_threshold": 0.99,
        "max_missing_rate": 0.5,
    }
    response = client.post("/api/v1/feature-engineering/select-features", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "const_feat" not in data["retained_features"]
    assert "feat1" in data["retained_features"]


def test_rank_importance_endpoint(client):
    payload = {
        "feature_data": [
            {"feat1": 1.0, "feat2": 100.0},
            {"feat1": 5.0, "feat2": 20.0},
            {"feat1": 10.0, "feat2": 5.0},
            {"feat1": 15.0, "feat2": 1.0},
        ],
        "target": [0, 0, 1, 1],
        "method": "random_forest",
        "top_k": 2,
    }
    response = client.post("/api/v1/feature-engineering/rank-importance", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["rankings"]) == 2
    assert "feature" in data["rankings"][0]


def test_pipeline_endpoint(client, sample_payload):
    payload = {
        "transactions": sample_payload["transactions"],
        "accounts": sample_payload["accounts"],
        "target_labels": {"acc_0": 0, "acc_1": 1, "acc_2": 0},
        "max_depth": 2,
        "top_k": 5,
    }
    response = client.post("/api/v1/feature-engineering/pipeline", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_features_generated"] > 0
    assert data["num_features_selected"] <= 5
