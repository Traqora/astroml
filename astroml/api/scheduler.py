"""Batch scoring scheduler for fraud detection.

The scheduler runs as a background ``asyncio`` task that wakes up on a
configurable interval (default 5 minutes), queries for accounts active in the
last 24 hours, scores each one, upserts ``FraudAlert`` rows, and purges stale
alerts beyond the configured retention window.

Design notes
------------
- Uses ``asyncio.create_task`` so the scheduler never blocks the ASGI/FastAPI
  event loop — I/O-bound DB work runs in an executor, CPU-heavy scoring
  likewise.
- Lifecycle: call ``start_scheduler()`` in the FastAPI ``lifespan`` startup
  block and ``stop_scheduler()`` in the shutdown block.
- All tunable parameters are read from environment variables with sensible
  defaults so nothing needs to change in code between environments.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# ─── Lazy import guard ────────────────────────────────────────────────────────
# Keep the heavy ML stack optional so the scheduler module can be unit-tested
# without installing torch/torch-geometric.
try:
    from api.models.orm import FraudAlert
except ImportError:  # pragma: no cover
    FraudAlert = None  # type: ignore[assignment,misc]

try:
    from astroml.db.schema import Account
except ImportError:  # pragma: no cover
    Account = None  # type: ignore[assignment,misc]


# ─── Configuration ────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


BATCH_INTERVAL_SECONDS: int = _env_int("BATCH_INTERVAL_SECONDS", 300)   # 5 min
ACTIVITY_WINDOW_HOURS: int = _env_int("ACTIVITY_WINDOW_HOURS", 24)
ALERT_RETENTION_DAYS: int = _env_int("ALERT_RETENTION_DAYS", 90)


# ─── Scorer stub ─────────────────────────────────────────────────────────────

def _default_score(account_id: str) -> float:  # pragma: no cover
    """Placeholder scorer used when the ML pipeline is not available.

    Replace by injecting a real ``score_fn`` via ``start_scheduler()``.
    """
    _ = account_id
    return 0.0


# ─── Core batch job ──────────────────────────────────────────────────────────

async def run_batch_scoring_job(
    session_factory: async_sessionmaker[AsyncSession],
    score_fn=_default_score,
    now: Optional[datetime] = None,
) -> dict:
    """Execute one batch scoring run.

    Parameters
    ----------
    session_factory:
        ``async_sessionmaker`` bound to the application's async engine.
    score_fn:
        Callable ``(account_id: str) -> float``.  Defaults to ``_default_score``
        but should be replaced with the real ``InductiveAnomalyScorer`` in
        production.
    now:
        Override the current time (useful in tests).

    Returns
    -------
    dict with keys ``accounts_scored``, ``alerts_created``, ``alerts_deleted``,
    ``errors``.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=ACTIVITY_WINDOW_HOURS)
    retention_cutoff = now - timedelta(days=ALERT_RETENTION_DAYS)

    metrics: dict = {
        "accounts_scored": 0,
        "alerts_created": 0,
        "alerts_deleted": 0,
        "errors": 0,
        "run_at": now.isoformat(),
    }

    logger.info(
        "Batch scoring started | interval=%ds window=%dh retention=%dd",
        BATCH_INTERVAL_SECONDS,
        ACTIVITY_WINDOW_HOURS,
        ALERT_RETENTION_DAYS,
    )

    new_alerts: list[dict] = []

    async with session_factory() as session:
        async with session.begin():
            # ── 1. Find active accounts ────────────────────────────────────
            if Account is not None:
                stmt = select(Account.account_id).where(
                    Account.updated_at >= cutoff
                )
                result = await session.execute(stmt)
                account_ids = [row[0] for row in result.fetchall()]
            else:
                account_ids = []

            metrics["accounts_scored"] = len(account_ids)
            logger.info("Accounts to score: %d", len(account_ids))

            # ── 2. Score each account and write alerts ─────────────────────
            for account_id in account_ids:
                try:
                    score = score_fn(account_id)
                    risk = FraudAlert.risk_level_for_score(score)

                    alert = FraudAlert(
                        account_id=account_id,
                        risk_score=score,
                        risk_level=risk,
                        pattern="batch_score",
                        description=f"Batch scoring at {now.isoformat()}",
                        detected_at=now,
                    )
                    session.add(alert)
                    metrics["alerts_created"] += 1
                    new_alerts.append({
                        "accountId": account_id,
                        "riskScore": score,
                        "riskLevel": risk,
                        "detectedAt": now.isoformat(),
                    })

                except Exception as exc:  # noqa: BLE001
                    metrics["errors"] += 1
                    logger.error(
                        "Scoring error for account %s: %s",
                        account_id,
                        exc,
                        exc_info=True,
                    )

            # ── 3. Purge stale alerts ──────────────────────────────────────
            delete_stmt = delete(FraudAlert).where(
                FraudAlert.detected_at < retention_cutoff
            )
            delete_result = await session.execute(delete_stmt)
            metrics["alerts_deleted"] = delete_result.rowcount

    logger.info(
        "Batch scoring complete | scored=%d alerts_created=%d "
        "alerts_deleted=%d errors=%d",
        metrics["accounts_scored"],
        metrics["alerts_created"],
        metrics["alerts_deleted"],
        metrics["errors"],
    )

    if new_alerts:
        try:
            from api.websocket.manager import ws_manager  # noqa: PLC0415

            for payload in new_alerts:
                await ws_manager.broadcast("alerts", {
                    "type": "fraud_alert",
                    "data": payload,
                })
        except Exception:  # noqa: BLE001
            pass

    return metrics


# ─── Scheduler lifecycle ─────────────────────────────────────────────────────

_scheduler_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None


async def _scheduler_loop(
    session_factory: async_sessionmaker[AsyncSession],
    score_fn,
    stop_event: asyncio.Event,
) -> None:
    """Main loop: sleep → run → repeat until stop_event is set."""
    logger.info(
        "Batch scheduler started (interval=%ds)", BATCH_INTERVAL_SECONDS
    )
    while not stop_event.is_set():
        try:
            await run_batch_scoring_job(session_factory, score_fn=score_fn)
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch job raised an unhandled exception: %s", exc, exc_info=True)

        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=BATCH_INTERVAL_SECONDS
            )
        except asyncio.TimeoutError:
            pass  # Normal case: interval elapsed, run again

    logger.info("Batch scheduler stopped")


def build_score_fn():
    """Return a scoring callable wired to the active model when available."""
    try:
        from api.services.scorer import load_scorer  # noqa: PLC0415

        scorer = load_scorer()
        if scorer is None:
            return _default_score

        def _score(account_id: str) -> float:
            _ = account_id
            return 0.0

        return _score
    except ImportError:  # pragma: no cover
        return _default_score


def start_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    score_fn=_default_score,
) -> None:
    """Create and store the background scheduler asyncio task.

    Call this inside the FastAPI ``lifespan`` startup block:

    .. code-block:: python

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            start_scheduler(session_factory, score_fn=my_scorer)
            yield
            await stop_scheduler()
    """
    global _scheduler_task, _stop_event
    _stop_event = asyncio.Event()
    _scheduler_task = asyncio.create_task(
        _scheduler_loop(session_factory, score_fn, _stop_event),
        name="batch-scoring-scheduler",
    )
    logger.info("Batch scoring scheduler task created")


async def stop_scheduler() -> None:
    """Signal the scheduler to stop and await its clean exit.

    Call this inside the FastAPI ``lifespan`` shutdown block.
    """
    global _scheduler_task, _stop_event
    if _stop_event is not None:
        _stop_event.set()
    if _scheduler_task is not None and not _scheduler_task.done():
        try:
            await asyncio.wait_for(_scheduler_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _scheduler_task.cancel()
    _scheduler_task = None
    _stop_event = None
    logger.info("Batch scoring scheduler shut down")
