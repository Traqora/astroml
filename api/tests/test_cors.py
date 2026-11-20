"""Tests for CORS middleware behaviour and config validation (issue #1248)."""

from __future__ import annotations

import pytest
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


class TestCORSMiddleware:
    """Verify that the CORS middleware honours the configured origins."""

    def test_allowed_origin_preflight(self, db_engine, db_session):
        """OPTIONS preflight from an allowed origin returns that origin in ACAO."""
        import os

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from api.app import app
        from api.config import settings
        from api.database import get_db, get_sync_db, reset_engines
        from fastapi.testclient import TestClient

        # Patch settings before the middleware reads them.
        original_origins = settings.cors_origins
        original_credentials = settings.cors_credentials
        settings.cors_origins = ["http://localhost:5173"]
        settings.cors_credentials = True

        # Re-register CORS middleware with the patched settings.
        from starlette.middleware.cors import CORSMiddleware as _CORS

        app.middleware_stack = None  # force rebuild
        # Rebuild the middleware stack by temporarily re-adding CORS.
        # The simplest approach: use a fresh TestClient which calls
        # app.build_middleware_stack() after we've patched settings.

        async_url = str(db_engine.url).replace("sqlite://", "sqlite+aiosqlite://")
        os.environ["DATABASE_URL"] = async_url
        reset_engines()

        async_engine = create_async_engine(
            async_url,
            connect_args={"check_same_thread": False},
        )
        AsyncSessionLocal = async_sessionmaker(
            bind=async_engine, expire_on_commit=False, class_=AsyncSession
        )

        async def _override_async_db():
            async with AsyncSessionLocal() as session:
                yield session

        def _override_db():
            yield db_session

        app.dependency_overrides[get_sync_db] = _override_db
        app.dependency_overrides[get_db] = _override_async_db

        try:
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
        finally:
            settings.cors_origins = original_origins
            settings.cors_credentials = original_credentials
            app.dependency_overrides.clear()
            async_engine.sync_engine.dispose()

    def test_disallowed_origin_preflight(self, db_engine, db_session):
        """OPTIONS preflight from a disallowed origin must not echo that origin."""
        import os

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from api.app import app
        from api.config import settings
        from api.database import get_db, get_sync_db, reset_engines
        from fastapi.testclient import TestClient

        original_origins = settings.cors_origins
        original_credentials = settings.cors_credentials
        settings.cors_origins = ["http://localhost:5173"]
        settings.cors_credentials = True

        app.middleware_stack = None

        async_url = str(db_engine.url).replace("sqlite://", "sqlite+aiosqlite://")
        os.environ["DATABASE_URL"] = async_url
        reset_engines()

        async_engine = create_async_engine(
            async_url,
            connect_args={"check_same_thread": False},
        )
        AsyncSessionLocal = async_sessionmaker(
            bind=async_engine, expire_on_commit=False, class_=AsyncSession
        )

        async def _override_async_db():
            async with AsyncSessionLocal() as session:
                yield session

        def _override_db():
            yield db_session

        app.dependency_overrides[get_sync_db] = _override_db
        app.dependency_overrides[get_db] = _override_async_db

        try:
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
        finally:
            settings.cors_origins = original_origins
            settings.cors_credentials = original_credentials
            app.dependency_overrides.clear()
            async_engine.sync_engine.dispose()
