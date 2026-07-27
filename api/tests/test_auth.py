"""Integration tests — authentication (issue #240)."""
from __future__ import annotations

import pytest

from api.auth.security import create_access_token, hash_password
from api.models.orm import User


@pytest.fixture()
def auth_client(client, db_session, monkeypatch):
    """TestClient with auth enabled and a seeded admin user."""
    monkeypatch.setenv("AUTH_ENABLED", "true")

    db_session.add(User(
        username="testadmin",
        hashed_password=hash_password("secret"),
        scopes=["admin", "read:transactions", "read:fraud", "write:loyalty"],
    ))
    db_session.commit()
    return client


@pytest.mark.xdist_group("api_auth")
class TestAuthentication:

    def test_unauthenticated_request_returns_401(self, auth_client):
        resp = auth_client.get("/api/v1/fraud/alerts")
        assert resp.status_code == 401

    def test_login_returns_jwt(self, auth_client):
        resp = auth_client.post("/api/v1/auth/login", json={
            "username": "testadmin",
            "password": "secret",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_authenticated_request_succeeds(self, auth_client):
        login = auth_client.post("/api/v1/auth/login", json={
            "username": "testadmin",
            "password": "secret",
        })
        token = login.json()["access_token"]
        resp = auth_client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_expired_token_returns_401(self, auth_client):
        from datetime import timedelta

        token = create_access_token(
            "testadmin",
            ["admin"],
            expires_delta=timedelta(seconds=-1),
        )
        resp = auth_client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401

    def test_health_is_public(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200
