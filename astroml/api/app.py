"""AstroML FastAPI application.

Entry point for the REST API.  The ``lifespan`` context manager starts the
batch scoring scheduler on startup and stops it gracefully on shutdown.

Usage
-----
    uvicorn astroml.api.app:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
DATABASE_URL            Async SQLAlchemy URL (e.g. postgresql+asyncpg://…).
BATCH_INTERVAL_SECONDS  How often the scorer runs (default 300 s / 5 min).
ACTIVITY_WINDOW_HOURS   Accounts active within this window are scored (default 24).
ALERT_RETENTION_DAYS    FraudAlert rows older than this are purged (default 90).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from astroml.api.routers import accounts, fraud
from astroml.api.scheduler import start_scheduler, stop_scheduler

# ─── Database setup ───────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://astroml:astroml@localhost/astroml",
)

_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
_session_factory: async_sessionmaker = async_sessionmaker(
    _engine, expire_on_commit=False
)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Start the batch scheduler on startup; stop it cleanly on shutdown."""
    start_scheduler(_session_factory)
    try:
        yield
    finally:
        await stop_scheduler()


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AstroML Fraud Detection API",
    version="0.1.0",
    description=(
        "REST API for AstroML fraud detection. "
        "Includes a background batch scoring scheduler that runs on a "
        "configurable interval."
    ),
    lifespan=lifespan,
)


app.include_router(accounts.router, tags=["accounts"])
app.include_router(fraud.router, tags=["fraud"])


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness check — returns 200 when the server is running."""
    return {"status": "ok"}


@app.get("/api/v1/fraud-alerts/stats", tags=["fraud"])
async def fraud_alert_stats() -> dict:
    """Return high-level stats about the fraud alert table.

    This is a placeholder route; a full implementation would query the DB.
    """
    return {
        "description": (
            "Fraud alert statistics endpoint. "
            "Connect a real DB query here once the schema is migrated."
        )
    }
