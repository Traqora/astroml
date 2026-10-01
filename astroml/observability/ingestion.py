"""Ingestion heartbeat check — is the pipeline still delivering data?

The ingestion state store records a ``last_processed_at`` timestamp every time a
ledger is processed (see :meth:`astroml.ingestion.state.IngestionState.record_processed`).
Comparing that timestamp against the wall clock answers the only question that
matters for data freshness: *how long has it been since anything landed?*

Grading (``docs/ingestion-monitoring.md``):

``OK``
    A ledger was processed within ``stale_threshold_seconds``.
``DEGRADED``
    Nothing for ``stale_threshold_seconds`` (default 300s) — or no timestamp is
    on record at all: a state file written before the heartbeat existed, or no
    ingestion has ever run against this store.
``FAIL``
    Nothing for ``fail_threshold_seconds`` (default ``2 x`` the stale
    threshold).

Thresholds default from this module's constants and can be overridden per call
or with ``INGESTION_STALE_THRESHOLD_SECONDS`` / ``INGESTION_FAIL_THRESHOLD_SECONDS``.

Consumers:

* ``GET /healthz/ingestion`` serves the :class:`CheckResult`.
* ``GET /metrics`` publishes ``astroml_ingestion_last_success_timestamp_seconds``
  and ``astroml_ingestion_staleness_seconds`` on every scrape, so the staleness
  alert keeps firing while ingestion is silent instead of freezing at the last
  value it saw.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Final

from astroml.ingestion.metrics import (
    INGESTION_LAST_SUCCESS_TIMESTAMP,
    INGESTION_STALENESS_SECONDS,
)
from astroml.observability.health import CheckResult, HealthStatus

logger = logging.getLogger("astroml.observability.ingestion")

#: Seconds of silence after which ingestion is reported as ``DEGRADED``.
DEFAULT_STALE_THRESHOLD_SECONDS: Final[float] = 300.0

#: ``fail_threshold_seconds`` defaults to this multiple of the stale threshold.
FAIL_THRESHOLD_MULTIPLIER: Final[float] = 2.0

#: Env overrides for the two thresholds.
STALE_THRESHOLD_ENV: Final[str] = "INGESTION_STALE_THRESHOLD_SECONDS"
FAIL_THRESHOLD_ENV: Final[str] = "INGESTION_FAIL_THRESHOLD_SECONDS"


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a heartbeat timestamp from the state store into aware UTC.

    Args:
        value: ISO-8601 string written by
            :func:`~astroml.ingestion.state.utc_now_iso`.

    Returns:
        An aware UTC ``datetime``, or ``None`` when the value is missing or
        unparseable — e.g. a legacy state file, or a hand-edited one. A naive
        value is read as UTC because that is what the writer emits.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def staleness_seconds(
    last_processed_at: str | None,
    *,
    now: datetime | float | None = None,
) -> float | None:
    """Age in seconds of the newest processed ledger.

    Args:
        last_processed_at: Heartbeat timestamp, or ``None`` when none is recorded.
        now: Override for the current time, as a ``datetime`` or Unix seconds.
            Defaults to the wall clock.

    Returns:
        Seconds since ``last_processed_at`` (clamped at ``0.0`` so a state file
        written by a host with a slightly fast clock cannot look like the
        future), or ``None`` when there is no usable timestamp.
    """
    recorded = parse_timestamp(last_processed_at)
    if recorded is None:
        return None
    if now is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(now, datetime):
        moment = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        moment = moment.astimezone(timezone.utc)
    else:
        moment = datetime.fromtimestamp(float(now), tz=timezone.utc)
    return max(0.0, (moment - recorded).total_seconds())


def _resolve_seconds(explicit: float | None, env_name: str, default: float) -> float:
    """Resolve a threshold: explicit argument, then env var, then default."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(env_name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("Ignoring non-numeric %s=%r; using %s", env_name, raw, default)
    return default


def _resolve_store(state_store: Any | None, state_path: str | None) -> Any:
    """Return the caller's store, or one pointed at the resolved state file."""
    if state_store is not None:
        return state_store
    from astroml.ingestion.state import StateStore  # noqa: PLC0415 - cycle guard

    return StateStore(state_path)


def check_ingestion_heartbeat(
    state_store: Any | None = None,
    *,
    state_path: str | None = None,
    now: datetime | float | None = None,
    stale_threshold_seconds: float | None = None,
    fail_threshold_seconds: float | None = None,
) -> CheckResult:
    """Grade data freshness from the ingestion heartbeat in the state store.

    Args:
        state_store: Store exposing ``path`` and ``load() -> IngestionState``.
            Defaults to a :class:`~astroml.ingestion.state.StateStore` at
            ``state_path`` (else ``INGESTION_STATE_FILE``, else the default
            location).
        state_path: Path used only when ``state_store`` is omitted.
        now: Override for the current time (tests).
        stale_threshold_seconds: Silence after which status is ``DEGRADED``.
            Defaults to :data:`DEFAULT_STALE_THRESHOLD_SECONDS`, or the env
            override.
        fail_threshold_seconds: Silence after which status is ``FAIL``. Defaults
            to ``FAIL_THRESHOLD_MULTIPLIER x`` the stale threshold, or the env
            override.

    Returns:
        A :class:`CheckResult` named ``"ingestion"``. A store that cannot be read,
        or one with no heartbeat, is reported as ``DEGRADED`` rather than raised:
        freshness that cannot be established is not freshness that has been
        established. ``FAIL`` is reserved for data that is demonstrably older
        than ``fail_threshold_seconds``.
    """
    stale_after = _resolve_seconds(
        stale_threshold_seconds, STALE_THRESHOLD_ENV, DEFAULT_STALE_THRESHOLD_SECONDS
    )
    fail_after = _resolve_seconds(
        fail_threshold_seconds, FAIL_THRESHOLD_ENV, stale_after * FAIL_THRESHOLD_MULTIPLIER
    )

    started = time.perf_counter()
    store = _resolve_store(state_store, state_path)
    path = str(getattr(store, "path", None) or state_path or "<unknown>")

    try:
        state = store.load()
    except Exception as exc:  # noqa: BLE001 - a probe must always answer
        return CheckResult(
            name="ingestion",
            status=HealthStatus.DEGRADED,
            details={
                "path": path,
                "error_type": type(exc).__name__,
                "staleness_seconds": None,
                "stale_threshold_seconds": stale_after,
                "fail_threshold_seconds": fail_after,
            },
            remediation=(
                f"The ingestion state store at {path} could not be read "
                f"({type(exc).__name__}: {exc}). Freshness cannot be established. "
                "Inspect the shared ingestion volume and restore the state file "
                "from backup if it is truncated."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    recorded_at = getattr(state, "last_processed_at", None)
    age = staleness_seconds(recorded_at, now=now)

    details: dict[str, Any] = {
        "path": path,
        "last_processed_at": recorded_at,
        "last_processed_ledger": getattr(state, "last_processed_ledger", None),
        "staleness_seconds": None if age is None else round(age, 3),
        "stale_threshold_seconds": stale_after,
        "fail_threshold_seconds": fail_after,
    }

    if age is None:
        return CheckResult(
            name="ingestion",
            status=HealthStatus.DEGRADED,
            details=details,
            remediation=(
                f"No ingestion heartbeat is recorded in {path}. Either no "
                "backfill or stream worker has run against this store yet, or "
                "the file predates the heartbeat field. Start a worker "
                "(`python -m astroml.ingestion.backfill ...`) and confirm "
                "INGESTION_STATE_FILE points at a volume shared with this process."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    if age >= fail_after:
        status = HealthStatus.FAIL
        remediation = (
            f"No ledger has been ingested for {age:.0f}s (last at {recorded_at}, "
            f"ledger {details['last_processed_ledger']}); the fail threshold is "
            f"{fail_after:.0f}s. Data served from this store is stale. Check the "
            "ingestion worker is running and not wedged on Horizon retries, rate "
            "limiting, or database writes."
        )
    elif age >= stale_after:
        status = HealthStatus.DEGRADED
        remediation = (
            f"No ledger has been ingested for {age:.0f}s (last at {recorded_at}), "
            f"past the {stale_after:.0f}s stale threshold but under the "
            f"{fail_after:.0f}s fail threshold. Investigate before it escalates: "
            "slow upstream, backed-up queue, or a worker that is alive but making "
            "no progress."
        )
    else:
        status = HealthStatus.OK
        remediation = ""

    return CheckResult(
        name="ingestion",
        status=status,
        details=details,
        remediation=remediation,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def update_ingestion_metrics(result: CheckResult) -> None:
    """Sample a heartbeat check result into the freshness gauges.

    Sampled on every ``/metrics`` scrape (mirroring the DB pool gauges), which is
    what lets ``astroml_ingestion_staleness_seconds`` keep growing while
    ingestion is silent instead of freezing at the last value it saw.

    With no heartbeat on record both gauges are set to ``NaN``: the series still
    exists so dashboards do not break, but it never satisfies a staleness
    comparison. That keeps deployments which intentionally run no ingestion quiet
    while still catching a worker that runs and then stops.

    Args:
        result: Outcome of :func:`check_ingestion_heartbeat`.
    """
    recorded_at = parse_timestamp(result.details.get("last_processed_at"))
    age = result.details.get("staleness_seconds")

    if recorded_at is None or age is None:
        INGESTION_LAST_SUCCESS_TIMESTAMP.set(float("nan"))
        INGESTION_STALENESS_SECONDS.set(float("nan"))
        return

    INGESTION_LAST_SUCCESS_TIMESTAMP.set(recorded_at.timestamp())
    INGESTION_STALENESS_SECONDS.set(float(age))


def refresh_ingestion_metrics(
    state_store: Any | None = None,
    *,
    state_path: str | None = None,
    now: datetime | float | None = None,
) -> CheckResult:
    """Check freshness and publish the gauges in one step.

    Returns:
        The :class:`CheckResult`, for callers that also serve it over HTTP.
    """
    result = check_ingestion_heartbeat(state_store, state_path=state_path, now=now)
    update_ingestion_metrics(result)
    return result
