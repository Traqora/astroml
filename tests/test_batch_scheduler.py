"""Unit tests for the batch scoring scheduler (issue #258).

All tests use an in-memory SQLite database via SQLAlchemy async so no
PostgreSQL or real ML models are required.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.models.orm import FraudAlert  # noqa: F401 — registers ORM on Base
from astroml.db.schema import Base as SchemaBase  # for accounts table
from astroml.api.scheduler import (
    ALERT_RETENTION_DAYS,
    ACTIVITY_WINDOW_HOURS,
    run_batch_scoring_job,
    start_scheduler,
    stop_scheduler,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        # Create API-layer tables (FraudAlert) and Stellar schema tables (accounts etc.)
        await conn.run_sync(SchemaBase.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# ─── FraudAlert model tests ──────────────────────────────────────────────────

class TestFraudAlertModel:
    def test_risk_level_low(self):
        assert FraudAlert.risk_level_for_score(0.0) == "low"
        assert FraudAlert.risk_level_for_score(0.49) == "low"

    def test_risk_level_medium(self):
        assert FraudAlert.risk_level_for_score(0.5) == "medium"
        assert FraudAlert.risk_level_for_score(0.79) == "medium"

    def test_risk_level_high(self):
        assert FraudAlert.risk_level_for_score(0.8) == "high"
        assert FraudAlert.risk_level_for_score(1.0) == "high"


# ─── Batch job tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRunBatchScoringJob:
    async def test_returns_metrics_dict(self, session_factory):
        """Job always returns a dict with the required metric keys."""
        metrics = await run_batch_scoring_job(session_factory)
        assert "accounts_scored" in metrics
        assert "alerts_created" in metrics
        assert "alerts_deleted" in metrics
        assert "errors" in metrics
        assert "run_at" in metrics

    async def test_creates_alert_for_each_scored_account(self, session_factory, engine):
        """One FraudAlert row is written per active account returned by the DB."""
        from astroml.db.schema import Account

        now = datetime.now(timezone.utc)

        # Insert two accounts updated within the activity window
        async with session_factory() as sess:
            async with sess.begin():
                for acct_id in ["GAAA000000000000000000000000000000000000000000000000001",
                                 "GAAA000000000000000000000000000000000000000000000000002"]:
                    sess.add(Account(
                        account_id=acct_id,
                        updated_at=now - timedelta(hours=1),
                    ))

        metrics = await run_batch_scoring_job(
            session_factory,
            score_fn=lambda _: 0.9,
            now=now,
        )

        assert metrics["accounts_scored"] == 2
        assert metrics["alerts_created"] == 2
        assert metrics["errors"] == 0

    async def test_no_accounts_yields_zero_alerts(self, session_factory):
        """When no accounts are active the job creates no alerts and reports 0 errors."""
        metrics = await run_batch_scoring_job(
            session_factory,
            score_fn=lambda _: 0.5,
        )
        assert metrics["accounts_scored"] == 0
        assert metrics["alerts_created"] == 0
        assert metrics["errors"] == 0

    async def test_scoring_error_increments_error_counter(self, session_factory, engine):
        """Exceptions raised by score_fn are caught and counted, not re-raised."""
        from astroml.db.schema import Account

        now = datetime.now(timezone.utc)
        acct_id = "GAAA000000000000000000000000000000000000000000000000ERR"

        async with session_factory() as sess:
            async with sess.begin():
                sess.add(Account(
                    account_id=acct_id,
                    updated_at=now - timedelta(hours=1),
                ))

        def boom(_account_id):
            raise RuntimeError("scorer exploded")

        metrics = await run_batch_scoring_job(session_factory, score_fn=boom, now=now)

        assert metrics["errors"] == 1
        assert metrics["alerts_created"] == 0

    async def test_old_alerts_are_purged(self, session_factory, engine):
        """Alerts older than ALERT_RETENTION_DAYS are deleted by the job."""
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(days=ALERT_RETENTION_DAYS + 1)

        # Insert a stale alert directly
        async with session_factory() as sess:
            async with sess.begin():
                stale = FraudAlert(
                    account_id="GAAA_OLD",
                    risk_score=0.1,
                    risk_level="low",
                    detected_at=stale_time,
                )
                sess.add(stale)

        metrics = await run_batch_scoring_job(
            session_factory,
            score_fn=lambda _: 0.0,
            now=now,
        )

        assert metrics["alerts_deleted"] >= 1

        # Verify the stale alert is gone
        async with session_factory() as sess:
            result = await sess.execute(
                select(FraudAlert).where(FraudAlert.account_id == "GAAA_OLD")
            )
            assert result.scalar_one_or_none() is None

    async def test_recent_alerts_are_not_purged(self, session_factory, engine):
        """Alerts within the retention window must not be deleted."""
        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(days=ALERT_RETENTION_DAYS - 1)

        async with session_factory() as sess:
            async with sess.begin():
                fresh = FraudAlert(
                    account_id="GAAA_NEW",
                    risk_score=0.7,
                    risk_level="medium",
                    detected_at=recent_time,
                )
                sess.add(fresh)

        await run_batch_scoring_job(
            session_factory,
            score_fn=lambda _: 0.0,
            now=now,
        )

        async with session_factory() as sess:
            result = await sess.execute(
                select(FraudAlert).where(FraudAlert.account_id == "GAAA_NEW")
            )
            assert result.scalar_one_or_none() is not None


# ─── Scheduler lifecycle tests ────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSchedulerLifecycle:
    async def test_start_and_stop_gracefully(self, session_factory):
        """start_scheduler creates a task; stop_scheduler cancels it cleanly."""
        start_scheduler(session_factory, score_fn=lambda _: 0.0)
        # Give the event loop one tick so the task starts
        await asyncio.sleep(0)
        await stop_scheduler()
        # Should not raise

    async def test_stop_is_idempotent(self):
        """Calling stop_scheduler when no scheduler is running is safe."""
        await stop_scheduler()  # no-op — should not raise

    async def test_scheduler_does_not_block_event_loop(self, session_factory):
        """The scheduler task yields back to the event loop between runs."""
        start_scheduler(session_factory, score_fn=lambda _: 0.0)
        # If the scheduler blocked the event loop this sleep would never fire
        done = asyncio.Event()

        async def set_done():
            await asyncio.sleep(0.05)
            done.set()

        asyncio.create_task(set_done())
        await asyncio.wait_for(done.wait(), timeout=2)
        assert done.is_set()
        await stop_scheduler()
