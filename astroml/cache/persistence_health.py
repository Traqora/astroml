"""Redis persistence health inspection (Issue #963).

Redis, in the default configuration used by :mod:`astroml.cache.redis_cache`,
keeps everything in memory only: a process restart or crash silently loses
every cached and durable value (including ``CacheKeyPrefix.INGESTION_STATE``,
which the ingestion pipeline relies on for resumability) unless RDB
snapshotting or AOF is enabled. This module turns Redis' own
``INFO persistence`` and ``CONFIG GET save`` output into a typed
:class:`PersistenceStats` snapshot with a policy on top, mirroring
:mod:`astroml.db.pool_health`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from astroml.observability.health import CheckResult, HealthStatus


@dataclass(frozen=True)
class PersistenceStats:
    """Snapshot of a Redis instance's persistence configuration.

    Attributes:
        rdb_enabled: True when at least one RDB save point (the ``save``
            directive) is configured.
        aof_enabled: True when append-only-file persistence is enabled.
        rdb_last_bgsave_status: ``"ok"`` or ``"err"`` per Redis' ``INFO``
            output (``"unknown"`` if the field is absent).
        aof_last_write_status: ``"ok"`` or ``"err"`` per Redis' ``INFO``
            output; only meaningful when ``aof_enabled`` is true.
        rdb_changes_since_last_save: Number of writes since the last
            successful RDB snapshot.
    """

    rdb_enabled: bool
    aof_enabled: bool
    rdb_last_bgsave_status: str
    aof_last_write_status: str
    rdb_changes_since_last_save: int

    @property
    def any_persistence_enabled(self) -> bool:
        """True when either RDB snapshots or AOF are configured."""
        return self.rdb_enabled or self.aof_enabled

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in a health-check JSON response."""
        return {
            "rdb_enabled": self.rdb_enabled,
            "aof_enabled": self.aof_enabled,
            "rdb_last_bgsave_status": self.rdb_last_bgsave_status,
            "aof_last_write_status": self.aof_last_write_status,
            "rdb_changes_since_last_save": self.rdb_changes_since_last_save,
        }


def collect_persistence_stats(client: Any) -> PersistenceStats:
    """Read persistence configuration and status off a redis-py client.

    Args:
        client: A ``redis.Redis``-compatible client (anything exposing
            ``.info()`` and ``.config_get()`` with the same signatures).

    Returns:
        A :class:`PersistenceStats` snapshot.
    """
    info = client.info(section="persistence") or {}
    save_config = client.config_get("save") or {}
    save_value = save_config.get("save", "")

    return PersistenceStats(
        rdb_enabled=bool(save_value),
        aof_enabled=bool(int(info.get("aof_enabled", 0))),
        rdb_last_bgsave_status=str(info.get("rdb_last_bgsave_status", "unknown")),
        aof_last_write_status=str(info.get("aof_last_write_status", "unknown")),
        rdb_changes_since_last_save=int(info.get("rdb_changes_since_last_save", 0)),
    )


def evaluate_persistence_health(stats: PersistenceStats) -> CheckResult:
    """Classify a persistence snapshot and attach remediation guidance.

    Args:
        stats: Persistence snapshot from :func:`collect_persistence_stats`.

    Returns:
        A :class:`CheckResult` named ``"redis_persistence"``. ``FAIL`` when
        neither RDB snapshots nor AOF are enabled (a restart silently loses
        everything); ``DEGRADED`` when persistence is enabled but the most
        recent write attempt failed.
    """
    if not stats.any_persistence_enabled:
        return CheckResult(
            name="redis_persistence",
            status=HealthStatus.FAIL,
            details=stats.to_dict(),
            remediation=(
                "Neither RDB snapshotting (`save`) nor AOF is configured on "
                "this Redis instance. A restart or crash will silently lose "
                "all cached and durable state. Enable `appendonly yes` or "
                "configure `save` points."
            ),
        )
    if stats.aof_enabled and stats.aof_last_write_status != "ok":
        return CheckResult(
            name="redis_persistence",
            status=HealthStatus.DEGRADED,
            details=stats.to_dict(),
            remediation=(
                "The last AOF write failed. Check disk space and "
                "permissions on the Redis data directory."
            ),
        )
    if stats.rdb_enabled and stats.rdb_last_bgsave_status != "ok":
        return CheckResult(
            name="redis_persistence",
            status=HealthStatus.DEGRADED,
            details=stats.to_dict(),
            remediation=(
                "The last RDB background save failed. Check disk space and "
                "permissions on the Redis data directory."
            ),
        )
    return CheckResult(
        name="redis_persistence",
        status=HealthStatus.OK,
        details=stats.to_dict(),
    )


def check_persistence(client: Any) -> CheckResult:
    """Collect and evaluate Redis persistence health in one call.

    Args:
        client: A ``redis.Redis``-compatible client.

    Returns:
        A :class:`CheckResult` named ``"redis_persistence"``, with
        ``duration_ms`` set.
    """
    started = time.perf_counter()
    result = evaluate_persistence_health(collect_persistence_stats(client))
    return CheckResult(
        name=result.name,
        status=result.status,
        details=result.details,
        remediation=result.remediation,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
