"""Tests for API versioning middleware and utilities (issue #572)."""
from __future__ import annotations

import pytest
from api.versioning import CURRENT_API_VERSION, LATEST_API_VERSION, SUPPORTED_VERSIONS, get_api_prefix


class TestVersionConstants:
    def test_current_version_is_defined(self):
        assert CURRENT_API_VERSION == "v1"

    def test_latest_version_is_defined(self):
        assert LATEST_API_VERSION == "1.0.0"

    def test_supported_versions_contains_v1(self):
        assert "v1" in SUPPORTED_VERSIONS


class TestGetApiPrefix:
    def test_returns_prefix_for_v1(self):
        assert get_api_prefix("v1") == "/api/v1"

    def test_raises_for_unsupported_version(self):
        with pytest.raises(ValueError, match="Unsupported API version"):
            get_api_prefix("v3")


class TestVersioningHeaders:
    def test_response_has_version_header(self, client):
        resp = client.get("/health")
        assert "X-API-Version" in resp.headers

    def test_version_header_value(self, client):
        resp = client.get("/health")
        assert resp.headers["X-API-Version"] == CURRENT_API_VERSION

    def test_response_has_latest_version_header(self, client):
        resp = client.get("/health")
        assert "X-API-Latest-Version" in resp.headers

    def test_latest_version_header_value(self, client):
        resp = client.get("/health")
        assert resp.headers["X-API-Latest-Version"] == LATEST_API_VERSION

    def test_api_root_returns_version(self, client):
        resp = client.get("/api/v1")
        data = resp.json()
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestRouterPrefixes:
    def test_v1_router_prefixes(self, client):
        resp = client.get("/api/v1")
        assert resp.status_code == 200

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
