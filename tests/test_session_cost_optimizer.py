"""Pool cost/sizing regression tests for astroml.db.session (issue #989)."""

import logging
import pathlib

import pytest
from pydantic import ValidationError

from astroml.db import session as session_mod
from astroml.db.session import DatabaseConfig


def test_defaults_bound_connection_count():
    cfg = DatabaseConfig()
    assert cfg.max_connections == cfg.pool_size + cfg.max_overflow == 30


def test_custom_max_connections():
    assert DatabaseConfig(pool_size=2, max_overflow=0).max_connections == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("pool_size", 0),
        ("pool_size", -5),
        ("max_overflow", -1),
        ("pool_timeout", 0),
        ("pool_recycle", -2),
    ],
)
def test_rejects_invalid_pool_settings(field, value):
    with pytest.raises(ValidationError):
        DatabaseConfig(**{field: value})


def test_pool_recycle_minus_one_allowed():
    assert DatabaseConfig(pool_recycle=-1).pool_recycle == -1


def test_invalid_yaml_pool_size_raises(tmp_path: pathlib.Path):
    path = tmp_path / "database.yaml"
    path.write_text("database:\n  pool_size: -1\n")
    with pytest.raises(ValueError, match="pool_size"):
        session_mod.load_database_config(path)


def test_get_engine_fallback_is_logged(monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise FileNotFoundError("no config")

    created = {}

    def fake_create_engine(url, **kwargs):
        created.update(kwargs)
        return object()

    monkeypatch.setattr(session_mod, "load_database_config", boom)
    monkeypatch.setattr(session_mod, "resolve_database_url", lambda: "postgresql://x")
    monkeypatch.setattr(session_mod, "create_engine", fake_create_engine)
    monkeypatch.setattr(session_mod, "_enable_query_profiling_if_debug", lambda e: None)
    session_mod.get_engine.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger=session_mod.logger.name):
            session_mod.get_engine()
    finally:
        session_mod.get_engine.cache_clear()

    assert created["pool_size"] == 10 and created["max_overflow"] == 20
    assert any("Falling back" in r.message for r in caplog.records)
