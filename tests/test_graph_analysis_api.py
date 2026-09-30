"""Tests for Graph Analysis API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.graph_analysis import router


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
                "source_account": "G_ACC_1",
                "destination_account": "G_ACC_2",
                "amount": 150.0,
                "fee": 0.01,
                "asset": "XLM",
            },
            {
                "source_account": "G_ACC_2",
                "destination_account": "G_ACC_3",
                "amount": 75.0,
                "fee": 0.01,
                "asset": "XLM",
            },
            {
                "source_account": "G_ACC_3",
                "destination_account": "G_ACC_1",
                "amount": 25.0,
                "fee": 0.01,
                "asset": "XLM",
            },
            {
                "source_account": "G_ACC_1",
                "destination_account": "G_ACC_4",
                "amount": 500.0,
                "fee": 0.05,
                "asset": "USD",
            },
        ]
    }


def test_health_check(client):
    response = client.get("/api/v1/graph-analysis/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_graph(client, sample_payload):
    response = client.post("/api/v1/graph-analysis/analyze", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_nodes"] == 4
    assert data["num_edges"] == 4
    assert "density" in data
    assert "avg_degree" in data


def test_classify_nodes_graphsage(client, sample_payload):
    payload = {
        **sample_payload,
        "model_type": "graphsage",
        "hidden_dim": 16,
        "num_classes": 2,
    }
    response = client.post("/api/v1/graph-analysis/classify-nodes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_nodes"] == 4
    assert len(data["predictions"]) == 4
    for pred in data["predictions"]:
        assert "account" in pred
        assert "fraud_probability" in pred
        assert "risk_score" in pred
        assert 0.0 <= pred["fraud_probability"] <= 1.0


def test_classify_nodes_gat(client, sample_payload):
    payload = {
        **sample_payload,
        "model_type": "gat",
        "hidden_dim": 16,
        "num_classes": 2,
    }
    response = client.post("/api/v1/graph-analysis/classify-nodes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 4


def test_embeddings_endpoint(client, sample_payload):
    payload = {
        **sample_payload,
        "model_type": "graphsage",
        "embedding_dim": 16,
    }
    response = client.post("/api/v1/graph-analysis/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 4
    for acc, emb in data["embeddings"].items():
        assert len(emb) == 16


def test_subgraph_endpoint(client, sample_payload):
    payload = {
        **sample_payload,
        "target_accounts": ["G_ACC_1"],
        "num_hops": 1,
    }
    response = client.post("/api/v1/graph-analysis/subgraph", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "G_ACC_1" in data["target_accounts"]
    assert len(data["subgraph_nodes"]) >= 1


def test_train_gnn_endpoint(client, sample_payload):
    payload = {
        **sample_payload,
        "labels": {
            "G_ACC_1": 0,
            "G_ACC_2": 1,
            "G_ACC_3": 0,
            "G_ACC_4": 1,
        },
        "model_type": "graphsage",
        "epochs": 10,
        "lr": 0.01,
        "hidden_dim": 16,
    }
    response = client.post("/api/v1/graph-analysis/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["epochs_trained"] == 10
    assert data["final_train_loss"] >= 0.0


def test_empty_transactions_error(client):
    response = client.post("/api/v1/graph-analysis/analyze", json={"transactions": []})
    assert response.status_code == 400
