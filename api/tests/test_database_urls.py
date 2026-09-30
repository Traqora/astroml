"""
Unit tests — DATABASE_URL driver normalisation.

Covers: `_async_url()` yields a scheme `create_async_engine` accepts for every
sync DSN this repo's compose/k8s/CI configs publish, and `_sync_url()` stays its
inverse.
"""

from __future__ import annotations

import pytest

from api.database import _async_url, _sync_url

SYNC_TO_ASYNC = [
    (
        "postgresql://astroml:pw@postgres:5432/astroml",
        "postgresql+asyncpg://astroml:pw@postgres:5432/astroml",
    ),
    (
        "postgres://astroml:pw@postgres:5432/astroml",
        "postgresql+asyncpg://astroml:pw@postgres:5432/astroml",
    ),
    (
        "postgresql+psycopg2://astroml:pw@postgres:5432/astroml",
        "postgresql+asyncpg://astroml:pw@postgres:5432/astroml",
    ),
    ("sqlite:///./astroml.db", "sqlite+aiosqlite:///./astroml.db"),
]

ALREADY_ASYNC = [
    "postgresql+asyncpg://astroml:pw@postgres:5432/astroml",
    "sqlite+aiosqlite:///./astroml.db",
]


class TestAsyncUrl:
    @pytest.mark.parametrize("given,expected", SYNC_TO_ASYNC)
    def test_upgrades_sync_schemes(self, monkeypatch, given, expected):
        monkeypatch.setenv("DATABASE_URL", given)
        assert _async_url() == expected

    @pytest.mark.parametrize("url", ALREADY_ASYNC)
    def test_leaves_async_schemes_alone(self, monkeypatch, url):
        monkeypatch.setenv("DATABASE_URL", url)
        assert _async_url() == url

    def test_credentials_and_query_string_survive(self, monkeypatch):
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://u:p%40ss@db.internal:6543/astroml?sslmode=require",
        )
        assert (
            _async_url() == "postgresql+asyncpg://u:p%40ss@db.internal:6543/astroml?sslmode=require"
        )

    def test_default_is_already_async(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert "+asyncpg" in _async_url()


class TestSyncUrlStaysInverse:
    @pytest.mark.parametrize("given,_expected", SYNC_TO_ASYNC)
    def test_sync_url_never_carries_an_async_driver(self, monkeypatch, given, _expected):
        monkeypatch.setenv("DATABASE_URL", given)
        assert "+asyncpg" not in _sync_url()
        assert "+aiosqlite" not in _sync_url()

    @pytest.mark.parametrize("url", ALREADY_ASYNC)
    def test_one_env_var_feeds_both_engines(self, monkeypatch, url):
        """The two views may differ only in the driver they name."""
        monkeypatch.setenv("DATABASE_URL", url)
        assert _async_url() == url
        assert _sync_url() == url.replace("+asyncpg", "").replace("+aiosqlite", "")


class TestSqlalchemyAcceptsTheUrl:
    """The real contract is SQLAlchemy's, not a string comparison."""

    @pytest.mark.parametrize(
        "given",
        [url for url, _ in SYNC_TO_ASYNC] + ALREADY_ASYNC,
    )
    def test_every_published_dsn_builds_an_async_engine(self, monkeypatch, given):
        pytest.importorskip("asyncpg")
        pytest.importorskip("aiosqlite")
        from sqlalchemy.ext.asyncio import create_async_engine

        monkeypatch.setenv("DATABASE_URL", given)
        engine = create_async_engine(_async_url())
        try:
            assert engine.dialect.is_async is True
        finally:
            # AsyncEngine.dispose() is a coroutine; the sync pool is the part
            # that actually owns the (never-opened) connections here.
            engine.sync_engine.dispose()
