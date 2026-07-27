"""Tests for the API tracing configuration."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.instrumentation.fastapi")
pytest.importorskip("opentelemetry.instrumentation.httpx")
pytest.importorskip("opentelemetry.instrumentation.sqlalchemy")

from api import tracing


def test_build_resource_includes_environment(monkeypatch) -> None:
    monkeypatch.setenv("ENV", "staging")
    resource = tracing._build_resource()

    assert resource.attributes["service.name"] == tracing.settings.service_name
    assert resource.attributes["service.version"] == tracing.settings.api_version
    assert resource.attributes["deployment.environment"] == "staging"


def test_setup_tracing_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(tracing.settings, "tracing_enabled", False)
    assert tracing.setup_tracing() is None
