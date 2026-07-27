"""
Integration tests — health check endpoint (issue #244).

Covers: /health returns 200 with expected payload.
"""
from __future__ import annotations

import pytest


@pytest.mark.xdist_group("api_health")
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_is_json(self, client):
        resp = client.get("/health")
        assert resp.headers["content-type"].startswith("application/json")

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] in {"ok", "healthy", "up"}
