"""Tests for astroml.db.session's engine-config error handling (issue #970).

``get_engine()`` previously swallowed *any* exception raised while loading
``config/database.yaml`` silently before falling back to default pool
settings, so a malformed config file looked identical to "there's no config
file in this environment" — no trace of what actually went wrong ended up
in the logs. This asserts the fallback still happens (existing, relied-upon
behavior for environments without a config file) but is now observable via
a logged warning that carries the original error.
"""

from __future__ import annotations

import logging

import pytest

from astroml.db import session as session_module


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    """get_engine() is process-wide cached; isolate tests from each other."""
    session_module.get_engine.cache_clear()
    yield
    session_module.get_engine.cache_clear()


def test_get_engine_logs_warning_when_config_load_fails(monkeypatch, caplog):
    def _boom():
        raise ValueError("config/database.yaml is missing the `database:` key")

    monkeypatch.setattr(session_module, "load_database_config", _boom)
    monkeypatch.setattr(
        session_module, "resolve_database_url", lambda: "postgresql://u:p@localhost:5432/db"
    )
    monkeypatch.setattr(session_module, "create_engine", lambda *a, **k: object())

    with caplog.at_level(logging.WARNING, logger="astroml.db.session"):
        engine = session_module.get_engine()

    assert engine is not None
    assert any("Failed to load database config" in record.message for record in caplog.records)
    assert any("missing the `database:` key" in record.message for record in caplog.records)


def test_get_engine_still_falls_back_to_default_pool_settings(monkeypatch):
    calls = []

    def _boom():
        raise RuntimeError("boom")

    def _fake_create_engine(url, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(session_module, "load_database_config", _boom)
    monkeypatch.setattr(
        session_module, "resolve_database_url", lambda: "postgresql://u:p@localhost:5432/db"
    )
    monkeypatch.setattr(session_module, "create_engine", _fake_create_engine)

    session_module.get_engine()

    assert len(calls) == 1
    assert calls[0]["pool_size"] == 10
    assert calls[0]["max_overflow"] == 20
    assert calls[0]["pool_timeout"] == 30
    assert calls[0]["pool_recycle"] == 1800


def test_get_engine_does_not_log_warning_on_success(monkeypatch, caplog):
    """No config-load failure -> no warning; only the failure path is noisy."""
    from astroml.db.session import DatabaseConfig

    monkeypatch.setattr(session_module, "load_database_config", lambda: DatabaseConfig())
    monkeypatch.setattr(
        session_module, "resolve_database_url", lambda: "postgresql://u:p@localhost:5432/db"
    )
    monkeypatch.setattr(session_module, "create_engine", lambda *a, **k: object())

    with caplog.at_level(logging.WARNING, logger="astroml.db.session"):
        session_module.get_engine()

    assert not any("Failed to load database config" in record.message for record in caplog.records)
