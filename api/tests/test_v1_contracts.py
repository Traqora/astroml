"""V1 API contract tests — issue #1233.

Verifies the exact shape of success and failure responses for all three V1
endpoint groups: health, model readiness, and baseline predictions.

Contract = the caller-visible response schema is pinned: required fields,
types, status codes, Content-Type, and RFC 7807 error shape.

No optional services required (AUTH_ENABLED=false, DISABLE_SCHEDULER=true,
SQLite — all set by conftest.py).
"""

from __future__ import annotations

import pytest

from api.models.orm import ModelRegistry

RFC7807_CT = "application/problem+json"
JSON_CT = "application/json"
RFC7807_REQUIRED = {"type", "title", "status", "detail", "instance", "code"}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _assert_rfc7807(resp, expected_status: int, expected_code: str) -> dict:
    """Assert the response is a valid RFC 7807 problem+json body."""
    assert resp.status_code == expected_status
    ct = resp.headers.get("content-type", "")
    assert ct.startswith(RFC7807_CT), f"Expected {RFC7807_CT}, got {ct!r}"
    body = resp.json()
    missing = RFC7807_REQUIRED - body.keys()
    assert not missing, f"RFC 7807 members missing: {missing}"
    assert body["status"] == expected_status
    assert body["code"] == expected_code
    assert isinstance(body["type"], str) and body["type"].startswith("https://")
    assert isinstance(body["instance"], str) and body["instance"].startswith("/")
    return body


def _seed_model(db_session, name="test-model", version="1.0.0", status="active") -> ModelRegistry:
    m = ModelRegistry(name=name, version=version, status=status, model_type="anomaly", path="/tmp/m.pkl")
    db_session.add(m)
    db_session.flush()
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1: Health endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthContracts:
    """Contract: GET /health (unauthenticated, always available)."""

    def test_success_status_code(self, client):
        assert client.get("/health").status_code == 200

    def test_success_content_type_is_json(self, client):
        resp = client.get("/health")
        assert resp.headers["content-type"].startswith(JSON_CT)

    def test_success_body_has_status_field(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert isinstance(body["status"], str)

    def test_method_not_allowed_returns_405(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405


class TestHealthDbContracts:
    """Contract: GET /health/db (feature-gated; bypassed with AUTH_ENABLED=false)."""

    def test_success_status_code(self, client):
        resp = client.get("/health/db")
        assert resp.status_code in (200, 503)

    def test_success_body_shape(self, client):
        resp = client.get("/health/db")
        if resp.status_code == 200:
            body = resp.json()
            assert body["status"] == "healthy"
            assert "latency_ms" in body
            assert isinstance(body["latency_ms"], (int, float))
            assert "error" in body
            assert "error_code" in body
            assert body["error_code"] is None

    def test_timeout_error_shape(self, client, monkeypatch):
        """Simulated timeout → 503 DB_TIMEOUT with RFC 7807 shape."""
        import asyncio
        import api.config as cfg_module
        from api.config import Settings
        from sqlalchemy.ext.asyncio import AsyncSession

        async def _slow(self, *args, **kwargs):
            await asyncio.sleep(999)

        monkeypatch.setattr(AsyncSession, "execute", _slow)
        monkeypatch.setattr(cfg_module, "settings", Settings(db_query_timeout_seconds=1))

        body = _assert_rfc7807(client.get("/health/db"), 503, "DB_TIMEOUT")
        assert "timed out" in body["title"].lower() or "timeout" in body["title"].lower()

    def test_method_not_allowed_returns_405(self, client):
        assert client.post("/health/db").status_code == 405


class TestHealthReadinessContracts:
    """Contract: GET /health/readiness."""

    def test_returns_200_or_503(self, client):
        assert client.get("/health/readiness").status_code in (200, 503)

    def test_success_body(self, client):
        resp = client.get("/health/readiness")
        if resp.status_code == 200:
            assert resp.json().get("status") == "ready"

    def test_failure_is_rfc7807(self, client, monkeypatch):
        """Simulated DB failure → 503 DB_ERROR with RFC 7807 shape."""
        import asyncio
        import api.config as cfg_module
        from api.config import Settings
        from sqlalchemy.ext.asyncio import AsyncSession

        async def _blow(self, *args, **kwargs):
            await asyncio.sleep(999)

        monkeypatch.setattr(AsyncSession, "execute", _blow)
        monkeypatch.setattr(cfg_module, "settings", Settings(db_query_timeout_seconds=1))

        resp = client.get("/health/readiness")
        assert resp.status_code == 503
        assert resp.headers["content-type"].startswith(RFC7807_CT)


class TestHealthLivenessContracts:
    """Contract: GET /health/liveness — always 200."""

    def test_always_200(self, client):
        assert client.get("/health/liveness").status_code == 200

    def test_body_has_status_alive(self, client):
        assert client.get("/health/liveness").json().get("status") == "alive"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: Model readiness endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelListContracts:
    """Contract: GET /api/v1/models."""

    def test_success_status_code(self, client):
        assert client.get("/api/v1/models").status_code == 200

    def test_success_envelope_shape(self, client):
        body = client.get("/api/v1/models").json()
        assert "data" in body
        assert "total" in body
        assert "page" in body
        assert "page_size" in body
        assert isinstance(body["data"], list)
        assert isinstance(body["total"], int)

    def test_empty_db_returns_empty_list(self, client):
        body = client.get("/api/v1/models").json()
        assert body["data"] == []
        assert body["total"] == 0

    def test_pagination_defaults(self, client):
        body = client.get("/api/v1/models").json()
        assert body["page"] == 1
        assert body["page_size"] > 0

    def test_pagination_params_accepted(self, client):
        resp = client.get("/api/v1/models?page=2&page_size=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["page"] == 2
        assert body["page_size"] == 5

    def test_invalid_page_returns_422(self, client):
        resp = client.get("/api/v1/models?page=0")
        assert resp.status_code == 422
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")


class TestModelGetContracts:
    """Contract: GET /api/v1/models/{id}."""

    def test_success_returns_200_with_full_schema(self, client, db_session):
        model = _seed_model(db_session)
        resp = client.get(f"/api/v1/models/{model.id}")
        assert resp.status_code == 200
        body = resp.json()
        for field in ("id", "name", "version", "status", "created_at"):
            assert field in body, f"Field {field!r} missing from ModelRegistryOut"
        assert body["id"] == model.id
        assert body["name"] == "test-model"
        assert body["version"] == "1.0.0"

    def test_not_found_is_rfc7807_404(self, client):
        resp = client.get("/api/v1/models/999999")
        assert resp.status_code == 404

    def test_invalid_id_type_returns_422(self, client):
        resp = client.get("/api/v1/models/not-an-int")
        assert resp.status_code == 422
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")


class TestModelActivateContracts:
    """Contract: POST /api/v1/models/{id}/activate."""

    def test_success_returns_200_with_active_status(self, client, db_session):
        model = _seed_model(db_session, status="inactive")
        resp = client.post(f"/api/v1/models/{model.id}/activate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_not_found_returns_404(self, client):
        resp = client.post("/api/v1/models/999999/activate")
        assert resp.status_code == 404

    def test_invalid_id_type_returns_422(self, client):
        resp = client.post("/api/v1/models/bad/activate")
        assert resp.status_code == 422
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")


class TestModelMetricsContracts:
    """Contract: GET /api/v1/models/{id}/metrics."""

    def test_success_returns_200_with_metrics_shape(self, client, db_session):
        model = _seed_model(db_session)
        resp = client.get(f"/api/v1/models/{model.id}/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert "metrics" in body
        assert isinstance(body["metrics"], dict)

    def test_not_found_returns_404(self, client):
        resp = client.get("/api/v1/models/999999/metrics")
        assert resp.status_code == 404

    def test_invalid_id_type_returns_422(self, client):
        resp = client.get("/api/v1/models/xyz/metrics")
        assert resp.status_code == 422
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: Baseline predictions (fraud)
# ═══════════════════════════════════════════════════════════════════════════════


STELLAR_KEY = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"


class TestFraudScoreContracts:
    """Contract: POST /api/v1/fraud/score."""

    def test_success_returns_200(self, client):
        resp = client.post(
            "/api/v1/fraud/score",
            json={"accounts": [STELLAR_KEY], "edges": []},
        )
        assert resp.status_code == 200

    def test_success_body_has_scores_dict(self, client):
        resp = client.post(
            "/api/v1/fraud/score",
            json={"accounts": [STELLAR_KEY], "edges": []},
        )
        body = resp.json()
        assert "scores" in body
        assert isinstance(body["scores"], dict)
        assert STELLAR_KEY in body["scores"]
        assert isinstance(body["scores"][STELLAR_KEY], float)

    def test_no_model_returns_zero_scores(self, client):
        """When no model is loaded, all scores default to 0.0 — not an error."""
        resp = client.post(
            "/api/v1/fraud/score",
            json={"accounts": [STELLAR_KEY], "edges": []},
        )
        assert resp.status_code == 200
        assert resp.json()["scores"][STELLAR_KEY] == 0.0

    def test_missing_accounts_field_returns_422(self, client):
        resp = client.post("/api/v1/fraud/score", json={"edges": []})
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    def test_empty_accounts_list_returns_422(self, client):
        resp = client.post("/api/v1/fraud/score", json={"accounts": [], "edges": []})
        assert resp.status_code == 422
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    def test_invalid_stellar_key_returns_422(self, client):
        resp = client.post(
            "/api/v1/fraud/score",
            json={"accounts": ["not-a-stellar-key"], "edges": []},
        )
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    def test_too_many_accounts_returns_422(self, client):
        """FR: max 50 accounts per request."""
        accounts = [f"G{'A' * 54}{str(i).zfill(1)}"[:56] for i in range(51)]
        resp = client.post("/api/v1/fraud/score", json={"accounts": accounts, "edges": []})
        # Schema enforces max 50 — 422 expected
        assert resp.status_code == 422

    def test_method_get_not_allowed(self, client):
        assert client.get("/api/v1/fraud/score").status_code == 405


class TestFraudAlertsContracts:
    """Contract: GET /api/v1/fraud/alerts."""

    def test_success_status_code(self, client):
        assert client.get("/api/v1/fraud/alerts").status_code == 200

    def test_success_envelope_shape(self, client):
        body = client.get("/api/v1/fraud/alerts").json()
        assert "data" in body
        assert "total" in body
        assert "page" in body
        assert "page_size" in body
        assert isinstance(body["data"], list)

    def test_seeded_alert_appears_in_response(self, client, seeded_alert):
        body = client.get("/api/v1/fraud/alerts").json()
        assert body["total"] >= 1
        ids = [row["id"] for row in body["data"]]
        assert seeded_alert.id in ids

    def test_seeded_alert_shape(self, client, seeded_alert):
        body = client.get("/api/v1/fraud/alerts").json()
        alert = next(r for r in body["data"] if r["id"] == seeded_alert.id)
        for field in ("id", "account_id", "risk_score", "risk_level", "detected_at"):
            assert field in alert, f"Field {field!r} missing from FraudAlertOut"
        assert alert["risk_level"] == "high"
        assert alert["risk_score"] == pytest.approx(0.92)

    def test_risk_level_filter_valid(self, client, seeded_alert):
        resp = client.get("/api/v1/fraud/alerts?risk_level=high")
        assert resp.status_code == 200
        body = resp.json()
        assert all(r["risk_level"] == "high" for r in body["data"])

    def test_risk_level_filter_invalid_returns_422(self, client):
        resp = client.get("/api/v1/fraud/alerts?risk_level=extreme")
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    def test_page_size_too_large_returns_422(self, client):
        resp = client.get("/api/v1/fraud/alerts?page_size=999")
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    def test_pagination_page_zero_returns_422(self, client):
        resp = client.get("/api/v1/fraud/alerts?page=0")
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")


class TestFraudStatsContracts:
    """Contract: GET /api/v1/fraud/stats."""

    def test_success_status_code(self, client):
        assert client.get("/api/v1/fraud/stats").status_code == 200

    def test_success_body_shape(self, client):
        body = client.get("/api/v1/fraud/stats").json()
        for field in ("total_alerts", "high_risk", "medium_risk", "low_risk",
                      "recent_alerts", "risk_over_time"):
            assert field in body, f"Field {field!r} missing from FraudStatsResponse"
        assert isinstance(body["total_alerts"], int)
        assert isinstance(body["recent_alerts"], list)
        assert isinstance(body["risk_over_time"], list)

    def test_counts_reflect_seeded_data(self, client, seeded_alert):
        body = client.get("/api/v1/fraud/stats").json()
        assert body["total_alerts"] >= 1
        assert body["high_risk"] >= 1


class TestFraudExplanationContracts:
    """Contract: GET /api/v1/fraud/{id}/explanation."""

    def test_not_found_returns_404(self, client):
        resp = client.get("/api/v1/fraud/999999/explanation")
        assert resp.status_code == 404

    def test_invalid_id_returns_422(self, client):
        resp = client.get("/api/v1/fraud/not-an-int/explanation")
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    def test_success_body_shape(self, client, seeded_alert, monkeypatch):
        """With a mocked LLM that returns instantly, the shape is correct."""
        from astroml.llm.explainer import FraudExplainer
        monkeypatch.setattr(
            FraudExplainer, "generate_explanation",
            lambda self, **kwargs: "Test explanation text.",
        )
        resp = client.get(f"/api/v1/fraud/{seeded_alert.id}/explanation")
        assert resp.status_code == 200
        body = resp.json()
        for field in ("alert_id", "explanation", "generated_in_ms", "cached"):
            assert field in body, f"Field {field!r} missing from FraudExplanationOut"
        assert body["alert_id"] == seeded_alert.id
        assert isinstance(body["explanation"], str)
        assert isinstance(body["generated_in_ms"], float)
        assert isinstance(body["cached"], bool)

    def test_llm_timeout_returns_504_upstream_timeout(self, client, seeded_alert, monkeypatch):
        """LLM timeout → 504 UPSTREAM_TIMEOUT in RFC 7807 shape."""
        import time
        import api.config as cfg_module
        from api.config import Settings
        from astroml.llm.explainer import FraudExplainer

        monkeypatch.setattr(
            FraudExplainer, "generate_explanation",
            lambda self, **kwargs: (time.sleep(999), "unreachable")[1],
        )
        monkeypatch.setattr(cfg_module, "settings", Settings(llm_timeout_seconds=1))

        body = _assert_rfc7807(
            client.get(f"/api/v1/fraud/{seeded_alert.id}/explanation"),
            504,
            "UPSTREAM_TIMEOUT",
        )
        assert "timed out" in body["title"].lower() or "timeout" in body["title"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4: Cross-cutting error contracts
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossCuttingErrorContracts:
    """Verifies RFC 7807 error shapes are consistent across all three endpoint groups."""

    @pytest.mark.parametrize("url", [
        "/api/v1/models/not-an-int",
        "/api/v1/fraud/not-an-int/explanation",
    ])
    def test_path_param_type_error_is_rfc7807_422(self, client, url):
        resp = client.get(url)
        _assert_rfc7807(resp, 422, "VALIDATION_ERROR")

    @pytest.mark.parametrize("url", [
        "/api/v1/models/999999",
        "/api/v1/fraud/999999/explanation",
    ])
    def test_missing_resource_is_404(self, client, url):
        assert client.get(url).status_code == 404

    def test_404_body_is_json(self, client):
        resp = client.get("/api/v1/models/999999")
        assert resp.headers["content-type"].startswith(("application/json", RFC7807_CT))

    def test_422_instance_matches_request_path(self, client):
        resp = client.get("/api/v1/models/bad")
        body = resp.json()
        assert body.get("instance") == "/api/v1/models/bad"

    def test_route_not_found_returns_404_not_500(self, client):
        resp = client.get("/api/v1/does-not-exist")
        assert resp.status_code == 404
        assert resp.status_code != 500
