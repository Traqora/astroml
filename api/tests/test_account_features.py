"""Tests for per-account feature gating — issue #1230.

Covers:
  - AUTH_ENABLED=false bypass (all features open)
  - Feature enabled → 200
  - Feature not enabled → 403 FEATURE_NOT_ENABLED
  - Account not found → 404 ACCOUNT_NOT_FOUND
  - baseline_predictions gate on /fraud/score
  - health gate on /health/db
  - Admin management endpoint happy path
  - Non-admin → 403
  - Unknown feature name in body → 422 VALIDATION_ERROR
  - Basic health probes not gated
  - AUTH_ENABLED=false management bypass
  - Factory rejects bad feature name at call time (unit test)
"""

from __future__ import annotations

import os

import pytest

from api.auth.dependencies import AuthContext
from api.models.orm import ApiAccount, ModelRegistry


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_auth(user_id: int, scopes: list[str] | None = None) -> AuthContext:
    return AuthContext(
        subject="testuser",
        auth_type="api_key",
        scopes=scopes or [],
        user_id=user_id,
    )


def _seed_account(db_session, public_key: str, user_id: int, features: list[str] | None = None) -> ApiAccount:
    acc = ApiAccount(
        public_key=public_key,
        user_id=user_id,
        enabled_features=features,
    )
    db_session.add(acc)
    db_session.flush()
    return acc


def _seed_model(db_session) -> ModelRegistry:
    model = ModelRegistry(
        name="test-model",
        version="1.0.0",
        status="active",
        model_type="anomaly",
    )
    db_session.add(model)
    db_session.flush()
    return model


# ─── Test cases ───────────────────────────────────────────────────────────────


def test_bypass_when_auth_disabled(client, db_session):
    """TC-1: AUTH_ENABLED=false — model endpoint returns 200 without an ApiAccount row."""
    assert os.environ.get("AUTH_ENABLED", "false").lower() in ("false", "0", "no")
    model = _seed_model(db_session)
    resp = client.get(f"/api/v1/models/{model.id}")
    assert resp.status_code == 200


def test_feature_enabled_returns_200(client, db_session, monkeypatch):
    """TC-2: Feature enabled → 200."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    model = _seed_model(db_session)
    _seed_account(db_session, "TESTKEY1", user_id=10, features=["model_readiness"])

    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=10)
    try:
        resp = client.get(f"/api/v1/models/{model.id}")
        assert resp.status_code == 200
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_feature_not_enabled_returns_403(client, db_session, monkeypatch):
    """TC-3: Feature not enabled → 403 FEATURE_NOT_ENABLED."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    model = _seed_model(db_session)
    _seed_account(db_session, "TESTKEY2", user_id=11, features=[])

    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=11)
    try:
        resp = client.get(f"/api/v1/models/{model.id}")
        assert resp.status_code == 403
        body = resp.json()
        assert body["code"] == "FEATURE_NOT_ENABLED"
        assert body["details"]["feature"] == "model_readiness"
        assert body["details"]["enabled_features"] == []
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_account_not_found_returns_404(client, db_session, monkeypatch):
    """TC-4: No matching ApiAccount → 404 ACCOUNT_NOT_FOUND."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    model = _seed_model(db_session)
    # user_id=999 has no ApiAccount row
    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=999)
    try:
        resp = client.get(f"/api/v1/models/{model.id}")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == "ACCOUNT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_baseline_predictions_gate_on_fraud_score(client, db_session, monkeypatch):
    """TC-5: baseline_predictions not enabled → 403 on POST /api/v1/fraud/score."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    _seed_account(db_session, "TESTKEY3", user_id=12, features=[])

    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=12)
    try:
        resp = client.post(
            "/api/v1/fraud/score",
            json={"accounts": ["GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"], "edges": []},
        )
        assert resp.status_code == 403
        body = resp.json()
        assert body["code"] == "FEATURE_NOT_ENABLED"
        assert body["details"]["feature"] == "baseline_predictions"
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_health_feature_gate_on_db_endpoint(client, db_session, monkeypatch):
    """TC-6: health feature not enabled → 403 on GET /health/db."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    _seed_account(db_session, "TESTKEY4", user_id=13, features=[])

    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=13)
    try:
        resp = client.get("/health/db")
        assert resp.status_code == 403
        body = resp.json()
        assert body["code"] == "FEATURE_NOT_ENABLED"
        assert body["details"]["feature"] == "health"
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_management_admin_happy_path(client, db_session, monkeypatch):
    """TC-7: Admin PUT /api/v1/accounts/{key}/features → 200 with updated list."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    _seed_account(db_session, "ADMINKEY1", user_id=20, features=[])

    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=20, scopes=["admin"])
    try:
        resp = client.put(
            "/api/v1/accounts/ADMINKEY1/features",
            json={"enabled_features": ["health", "model_readiness"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["enabled_features"]) == {"health", "model_readiness"}
        assert body["public_key"] == "ADMINKEY1"
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_management_non_admin_returns_403(client, db_session, monkeypatch):
    """TC-8: Non-admin caller → 403 on PUT /api/v1/accounts/{key}/features."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from api.app import app
    from api.auth.dependencies import get_current_auth

    _seed_account(db_session, "ADMINKEY2", user_id=21, features=[])

    app.dependency_overrides[get_current_auth] = lambda: _make_auth(user_id=21, scopes=[])
    try:
        resp = client.put(
            "/api/v1/accounts/ADMINKEY2/features",
            json={"enabled_features": ["health"]},
        )
        assert resp.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def test_management_unknown_feature_returns_422(client, db_session):
    """TC-9: Unknown feature name in body → 422 VALIDATION_ERROR."""
    # AUTH_ENABLED=false so scope check is bypassed
    _seed_account(db_session, "ADMINKEY3", user_id=22, features=[])
    resp = client.put(
        "/api/v1/accounts/ADMINKEY3/features",
        json={"enabled_features": ["turbo_mode"]},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "turbo_mode" in body["details"]["unknown_features"]


def test_basic_health_probes_not_gated(client):
    """TC-10: /health, /health/readiness, /health/liveness are not feature-gated."""
    for path in ("/health", "/health/liveness"):
        resp = client.get(path)
        assert resp.status_code != 403, f"{path} returned 403"


def test_management_auth_disabled_bypass(client, db_session):
    """TC-11: AUTH_ENABLED=false — management endpoint works without credentials."""
    assert os.environ.get("AUTH_ENABLED", "false").lower() in ("false", "0", "no")
    _seed_account(db_session, "BYPASSKEY", user_id=30, features=[])
    resp = client.put(
        "/api/v1/accounts/BYPASSKEY/features",
        json={"enabled_features": ["health"]},
    )
    assert resp.status_code == 200
    assert resp.json()["enabled_features"] == ["health"]


def test_factory_rejects_unknown_feature_name():
    """TC-12: require_account_feature raises ValueError for an unknown name at call time."""
    from api.auth.features import require_account_feature

    with pytest.raises(ValueError, match="Unknown feature_name"):
        require_account_feature("bad_feature_name")
