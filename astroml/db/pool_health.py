"""Connection pool health inspection (Issue #550).

SQLAlchemy's ``QueuePool`` exposes counters but no policy: this module turns
those counters into a typed :class:`PoolStats` snapshot with a utilization
ratio, an alert threshold, and operator remediation text. The result feeds
both ``GET /metrics/db-pool`` and the ``/healthz/db`` probe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool

from astroml.observability.health import CheckResult, HealthStatus

#: Utilization ratio at or above which the pool is considered saturated.
POOL_UTILIZATION_ALERT_THRESHOLD: Final[float] = 0.80


@dataclass(frozen=True)
class PoolStats:
    """Snapshot of a SQLAlchemy connection pool.

    Attributes:
        pool_size: Configured steady-state pool size.
        checked_in: Idle connections currently held by the pool.
        checked_out: Connections currently leased to the application.
        overflow: Connections created beyond ``pool_size``. SQLAlchemy
            reports a negative number when overflow headroom is unused; it
            is clamped to ``0`` here.
        max_overflow: Configured overflow allowance.
        implementation: Pool class name, e.g. ``"QueuePool"``.
    """

    pool_size: int
    checked_in: int
    checked_out: int
    overflow: int
    max_overflow: int
    implementation: str

    @property
    def capacity(self) -> int:
        """Maximum simultaneous connections (``pool_size + max_overflow``)."""
        return self.pool_size + self.max_overflow

    @property
    def utilization(self) -> float:
        """Checked-out connections as a ratio of :attr:`capacity`."""
        if self.capacity <= 0:
            return 0.0
        return min(self.checked_out / self.capacity, 1.0)

    @property
    def saturated(self) -> bool:
        """True when utilization reaches the alert threshold."""
        return self.utilization >= POOL_UTILIZATION_ALERT_THRESHOLD

    @property
    def exhausted(self) -> bool:
        """True when every connection slot is leased out."""
        return self.capacity > 0 and self.checked_out >= self.capacity

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the ``/metrics/db-pool`` JSON response."""
        return {
            "implementation": self.implementation,
            "pool_size": self.pool_size,
            "checked_in": self.checked_in,
            "checked_out": self.checked_out,
            "overflow": self.overflow,
            "max_overflow": self.max_overflow,
            "capacity": self.capacity,
            "utilization": round(self.utilization, 4),
            "utilization_percent": round(self.utilization * 100, 2),
            "alert_threshold_percent": round(POOL_UTILIZATION_ALERT_THRESHOLD * 100, 2),
            "saturated": self.saturated,
            "exhausted": self.exhausted,
        }


def _int_attr(pool: Pool | Any, name: str) -> int:
    """Read an optional integer counter off a pool implementation.

    ``NullPool`` and ``StaticPool`` (used by SQLite in tests) do not
    implement the ``QueuePool`` counters; missing counters read as ``0``.
    """
    getter = getattr(pool, name, None)
    if getter is None:
        return 0
    try:
        value = getter() if callable(getter) else getter
    except Exception:  # noqa: BLE001 - counters must never break a probe
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def collect_pool_stats(engine: Engine | Any) -> PoolStats:
    """Read pool counters from a sync or async SQLAlchemy engine.

    Args:
        engine: An ``Engine``, ``AsyncEngine``, or any object exposing a
            ``pool`` attribute.

    Returns:
        A :class:`PoolStats` snapshot. Pools without counters (``NullPool``,
        ``StaticPool``) report zeros rather than raising.
    """
    pool = getattr(getattr(engine, "sync_engine", engine), "pool", engine)

    overflow = _int_attr(pool, "overflow")
    max_overflow = _int_attr(pool, "_max_overflow")

    return PoolStats(
        pool_size=_int_attr(pool, "size"),
        checked_in=_int_attr(pool, "checkedin"),
        checked_out=_int_attr(pool, "checkedout"),
        overflow=max(overflow, 0),
        max_overflow=max(max_overflow, 0),
        implementation=type(pool).__name__,
    )


def evaluate_pool_health(stats: PoolStats) -> CheckResult:
    """Classify a pool snapshot and attach remediation guidance.

    Args:
        stats: Pool snapshot from :func:`collect_pool_stats`.

    Returns:
        A :class:`CheckResult` named ``"db_pool"``. Saturation is reported
        as ``DEGRADED`` (the service still works); full exhaustion is
        ``FAIL`` because new requests will block until ``pool_timeout``.
    """
    if stats.exhausted:
        return CheckResult(
            name="db_pool",
            status=HealthStatus.FAIL,
            details=stats.to_dict(),
            remediation=(
                f"All {stats.capacity} connections are checked out. New "
                "requests will block until pool_timeout expires. Look for "
                "sessions that are never closed, raise DB_POOL_MAX_SIZE / "
                "DB_POOL_MAX_OVERFLOW, or shed load."
            ),
        )
    if stats.saturated:
        return CheckResult(
            name="db_pool",
            status=HealthStatus.DEGRADED,
            details=stats.to_dict(),
            remediation=(
                f"Pool utilization is {stats.utilization:.0%}, at or above "
                f"the {POOL_UTILIZATION_ALERT_THRESHOLD:.0%} alert "
                "threshold. Check for slow queries holding connections and "
                "consider raising DB_POOL_MAX_SIZE before it exhausts."
            ),
        )
    return CheckResult(
        name="db_pool",
        status=HealthStatus.OK,
        details=stats.to_dict(),
    )


def check_pool(engine: Engine | Any) -> CheckResult:
    """Collect and evaluate pool health in one call.

    Args:
        engine: Engine whose pool should be inspected.

    Returns:
        A :class:`CheckResult` named ``"db_pool"``, with ``duration_ms`` set.
    """
    started = time.perf_counter()
    result = evaluate_pool_health(collect_pool_stats(engine))
    return CheckResult(
        name=result.name,
        status=result.status,
        details=result.details,
        remediation=result.remediation,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
