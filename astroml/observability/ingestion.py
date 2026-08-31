"""Ingestion heartbeat / stale-data alerts (Issue #758).

Exposes a health check that reports when no new ledger has been ingested
for longer than a configurable threshold.  The check is transport-agnostic
so it can be reused by the FastAPI health router, CLI, or periodic probes.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Final

from astroml.ingestion.state import StateStore
from astroml.observability.health import CheckResult, HealthStatus

#: Default threshold at which ingestion is considered stale (seconds).
DEFAULT_STALE_THRESHOLD_SECONDS: Final[int] = 300

#: Threshold at which ingestion is considered critically stale (seconds).
#: Used when the caller does not supply an explicit fail threshold.
DEFAULT_FAIL_THRESHOLD_SECONDS: Final[int] = 600


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp produced by ``StateStore``.

    Args:
        value: ISO-8601 string, e.g. ``2024-01-01T00:00:00Z``.

    Returns:
        A timezone-aware UTC datetime, or ``None`` if parsing fails.
    """
    if not value:
        return None
    try:
        # StateStore writes ``datetime.utcnow().isoformat() + "Z"``.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def check_ingestion_heartbeat(
    state_store: StateStore,
    *,
    stale_threshold_seconds: float = DEFAULT_STALE_THRESHOLD_SECONDS,
    fail_threshold_seconds: float | None = None,
    now: datetime | None = None,
) -> CheckResult:
    """Check whether the ingestion pipeline has processed a ledger recently.

    Args:
        state_store: Store that tracks ``last_processed_at``.
        stale_threshold_seconds: Seconds without a new ledger before the
            check becomes ``DEGRADED``.
        fail_threshold_seconds: Seconds without a new ledger before the
            check becomes ``FAIL``. Defaults to twice the stale threshold.
        now: Optional reference time for tests. Defaults to UTC now.

    Returns:
        A :class:`CheckResult` named ``"ingestion_heartbeat"``.
    """
    started = time.perf_counter()
    fail_threshold = fail_threshold_seconds or stale_threshold_seconds * 2
    now = now or datetime.now(timezone.utc)

    try:
        state = state_store.load()
    except OSError as exc:
        return CheckResult(
            name="ingestion_heartbeat",
            status=HealthStatus.FAIL,
            details={"state_path": state_store.path, "error": str(exc)},
            remediation=(
                "Cannot read ingestion state. Verify the state file path "
                "and that the process user has read access."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    last_processed_at = _parse_timestamp(state.last_processed_at)

    if last_processed_at is None:
        return CheckResult(
            name="ingestion_heartbeat",
            status=HealthStatus.DEGRADED,
            details={
                "last_processed_ledger": state.last_processed_ledger,
                "last_processed_at": state.last_processed_at,
                "stale_threshold_seconds": stale_threshold_seconds,
                "fail_threshold_seconds": fail_threshold,
            },
            remediation=(
                "No ingestion timestamp has been recorded yet. "
                "Run an ingestion batch or verify the state store is being updated."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    # Ensure comparison is timezone-aware.
    if last_processed_at.tzinfo is None:
        last_processed_at = last_processed_at.replace(tzinfo=timezone.utc)

    elapsed_seconds = (now - last_processed_at).total_seconds()

    if elapsed_seconds >= fail_threshold:
        status = HealthStatus.FAIL
        remediation = (
            f"No ledger ingested for {elapsed_seconds:.0f}s "
            f"(fail threshold {fail_threshold:.0f}s). "
            "Investigate the ingestion worker, Horizon stream, and network connectivity."
        )
    elif elapsed_seconds >= stale_threshold_seconds:
        status = HealthStatus.DEGRADED
        remediation = (
            f"No ledger ingested for {elapsed_seconds:.0f}s "
            f"(stale threshold {stale_threshold_seconds:.0f}s). "
            "Check the ingestion worker logs for stalls or rate-limiting."
        )
    else:
        status = HealthStatus.OK
        remediation = ""

    return CheckResult(
        name="ingestion_heartbeat",
        status=status,
        details={
            "last_processed_ledger": state.last_processed_ledger,
            "last_processed_at": state.last_processed_at,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "stale_threshold_seconds": stale_threshold_seconds,
            "fail_threshold_seconds": fail_threshold,
        },
        remediation=remediation,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
