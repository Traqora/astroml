"""Tests for CORS middleware behaviour and config validation (issue #1248)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import ValidationError


# ─── Config-level tests ───────────────────────────────────────────────────────


class TestCORSConfig:
    """Validate Settings-level CORS guard rules."""

    def test_wildcard_with_credentials_raises(self):
        """cors_origins=['*'] + cors_credentials=True must be rejected."""
        from api.config import Settings

        with pytest.raises((ValidationError, ValueError)):
            Settings(cors_origins=["*"], cors_credentials=True)

    def test_wildcard_without_credentials_succeeds(self):
        """cors_origins=['*'] + cors_credentials=False is a valid combination."""
        from api.config import Settings

        s = Settings(cors_origins=["*"], cors_credentials=False)
        assert "*" in s.cors_origins
        assert s.cors_credentials is False


# ─── Middleware tests ─────────────────────────────────────────────────────────


def _make_cors_app(origins: list[str], credentials: bool) -> FastAPI:
    """Return a minimal FastAPI app with CORSMiddleware configured as given."""
    mini = FastAPI()

    mini.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=credentials,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @mini.get("/health")
    async def _health():
        return {"status": "ok"}

    return mini


class TestCORSMiddleware:
    """Verify that the CORS middleware honours the configured origins."""

    def test_allowed_origin_preflight(self):
        """OPTIONS preflight from an allowed origin returns that origin in ACAO."""
        app = _make_cors_app(["http://localhost:5173"], credentials=False)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.options(
                "/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
        acao = response.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:5173", (
            f"Expected ACAO header 'http://localhost:5173', got {acao!r}"
        )

    def test_disallowed_origin_preflight(self):
        """OPTIONS preflight from a disallowed origin must not echo that origin."""
        app = _make_cors_app(["http://localhost:5173"], credentials=False)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.options(
                "/health",
                headers={
                    "Origin": "https://evil.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
        acao = response.headers.get("access-control-allow-origin", "")
        assert acao != "https://evil.example.com", (
            "Disallowed origin must not appear in Access-Control-Allow-Origin"
        )
