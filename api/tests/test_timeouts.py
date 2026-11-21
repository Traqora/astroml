"""Tests for V1 request timeout behaviour — issue #1231.

Covers:
  - DB timeout fires on /health/db → 503 DB_TIMEOUT
  - LLM timeout fires on /api/v1/fraud/{id}/explanation → 504 UPSTREAM_TIMEOUT
  - /health/db happy path → 200 healthy
  - /api/v1/fraud/alerts happy path → 200
"""

from __future__ import annotations

import asyncio
import time

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import api.config as cfg_module
from api.config import Settings
from astroml.llm.explainer import FraudExplainer


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _override_settings(monkeypatch, **kwargs):
    """Replace the module-level settings singleton for the duration of the test."""
    monkeypatch.setattr(cfg_module, "settings", Settings(**kwargs))


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_db_timeout_health_endpoint(client, monkeypatch):
    """AsyncSession.execute sleeping beyond timeout → 503 DB_TIMEOUT."""

    async def _slow_execute(self, *args, **kwargs):
        await asyncio.sleep(999)

    monkeypatch.setattr(AsyncSession, "execute", _slow_execute)
    _override_settings(monkeypatch, db_query_timeout_seconds=1)

    resp = client.get("/health/db")
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "DB_TIMEOUT"
    assert resp.headers["content-type"].startswith("application/problem+json")


def test_llm_timeout_explanation_endpoint(client, db_session, seeded_alert, monkeypatch):
    """LLM generate_explanation sleeping beyond timeout → 504 UPSTREAM_TIMEOUT."""

    def _slow_generate(self, *args, **kwargs):
        time.sleep(999)
        return "never reached"

    monkeypatch.setattr(FraudExplainer, "generate_explanation", _slow_generate)
    _override_settings(monkeypatch, llm_timeout_seconds=1)

    resp = client.get(f"/api/v1/fraud/{seeded_alert.id}/explanation")
    assert resp.status_code == 504
    body = resp.json()
    assert body["code"] == "UPSTREAM_TIMEOUT"
    assert resp.headers["content-type"].startswith("application/problem+json")


def test_health_happy_path(client):
    """/health/db completes instantly on SQLite → 200 healthy."""
    resp = client.get("/health/db")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "healthy"
    assert "DB_TIMEOUT" not in resp.text


def test_fraud_alerts_happy_path(client):
    """/api/v1/fraud/alerts returns 200 with no timeout errors."""
    resp = client.get("/api/v1/fraud/alerts")
    assert resp.status_code == 200
    assert "DB_TIMEOUT" not in resp.text
