"""V1 API smoke tests — issue #1232.

Verifies that every documented V1 endpoint responds at all, without requiring
optional services (Redis, MLflow, LLM provider). AUTH_ENABLED=false (the
conftest default) so feature gating is bypassed — these tests confirm
routing and basic response shapes, not auth/feature logic.
"""

from __future__ import annotations


def test_health_basic(client):
    """GET /health always returns 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("ok", "healthy")


def test_health_liveness(client):
    """GET /health/liveness returns 200."""
    resp = client.get("/health/liveness")
    assert resp.status_code == 200
    assert resp.json().get("status") == "alive"


def test_health_readiness(client):
    """GET /health/readiness returns 200 or 503 — not a 404 or 500."""
    resp = client.get("/health/readiness")
    assert resp.status_code in (200, 503)


def test_health_db(client):
    """GET /health/db (feature-gated; bypassed with AUTH_ENABLED=false) returns 200 or 503."""
    resp = client.get("/health/db")
    assert resp.status_code in (200, 503)
    if resp.status_code == 503:
        body = resp.json()
        assert "code" in body
        assert body["code"] in ("DB_TIMEOUT", "DB_ERROR")


def test_models_list(client):
    """GET /api/v1/models returns 200 with a paginated envelope."""
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body or isinstance(body, list)


def test_models_get_missing(client):
    """GET /api/v1/models/999999 returns 404 when the model does not exist."""
    resp = client.get("/api/v1/models/999999")
    assert resp.status_code == 404


def test_fraud_score_no_model(client):
    """POST /api/v1/fraud/score returns 200 with zero scores when no model is loaded."""
    resp = client.post(
        "/api/v1/fraud/score",
        json={
            "accounts": ["GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"],
            "edges": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "scores" in body
    assert body["scores"]["GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"] == 0.0


def test_fraud_score_invalid_body(client):
    """POST /api/v1/fraud/score with a missing required field returns 422."""
    resp = client.post("/api/v1/fraud/score", json={})
    assert resp.status_code == 422


def test_fraud_alerts(client):
    """GET /api/v1/fraud/alerts returns 200 with a paginated envelope."""
    resp = client.get("/api/v1/fraud/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert "total" in body


def test_fraud_alerts_invalid_risk_level(client):
    """GET /api/v1/fraud/alerts with an invalid risk_level returns 422."""
    resp = client.get("/api/v1/fraud/alerts?risk_level=extreme")
    assert resp.status_code == 422
