"""Redis persistence health checks (issues #956 and #963).

``RedisCache`` treats Redis as a durable-enough store for feature/prediction
caches, but a server with both RDB snapshotting and AOF disabled loses its
entire dataset on restart or failover, silently degrading every cache
consumer to a 100% miss rate.

This module provides two complementary interfaces:

* :func:`check_persistence_health` returns a :class:`PersistenceHealth`
  snapshot with human-readable issues and a strict error for an
  unqueryable server (issue #956).
* :func:`check_persistence` wraps :class:`PersistenceStats` and the
  ``redis_persistence`` health policy used by the observability layer,
  mirroring :mod:`astroml.db.pool_health` (issue #963).

Both read Redis's own ``CONFIG GET``/``INFO persistence`` output, so they
work against remote or managed instances without local disk access.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import redis

from astroml.observability.health import CheckResult, HealthStatus

logger = logging.getLogger(__name__)


@dataclass
class PersistenceHealth:
    """Result of a Redis persistence health check.

    Attributes:
        healthy: True when at least one persistence mechanism (RDB or AOF)
            is enabled and the server reports no recent save/rewrite failure.
        rdb_enabled: True when ``save`` points is non-empty (RDB
            snapshotting configured).
        aof_enabled: True when ``appendonly`` is ``yes``.
        last_bgsave_status: Redis's own ``rdb_last_bgsave_status`` field
            (``"ok"`` or ``"err"``), or ``None`` if the server never
            reported a bgsave (fresh instance with no save yet).
        last_aof_rewrite_status: Redis's own
            ``aof_last_bgrewrite_status`` field, or ``None`` when AOF is
            disabled (the field is absent from ``INFO persistence`` in
            that case).
        issues: Human-readable reasons ``healthy`` is False. Empty when
            healthy.
    """

    healthy: bool
    rdb_enabled: bool
    aof_enabled: bool
    last_bgsave_status: str | None
    last_aof_rewrite_status: str | None
    issues: list[str] = field(default_factory=list)


class PersistenceCheckError(RuntimeError):
    """Raised when the persistence check itself cannot run.

    Distinct from ``PersistenceHealth(healthy=False)``: this means the
    check couldn't ask Redis at all (connection failure, permission
    error), not that it asked and got an unhealthy answer.
    """


def check_persistence_health(client: redis.Redis) -> PersistenceHealth:
    """Check whether ``client``'s Redis server has persistence enabled.

    Args:
        client: A connected ``redis.Redis`` client. Callers typically
            pass ``RedisCache()._client`` (see
            ``astroml.cache.redis_cache.RedisCache``), but this function
            takes a plain client so it has no dependency on the
            ``RedisCache`` singleton and can be unit tested against a
            fake/mock client directly.

    Returns:
        A ``PersistenceHealth`` describing the current state.

    Raises:
        PersistenceCheckError: if the ``CONFIG GET``/``INFO`` calls
            themselves fail (connection lost, ``CONFIG`` command
            disabled on a managed instance, etc.). Distinguishing this
            from "checked and unhealthy" matters for alerting: a check
            that can't run should page differently than a check that
            ran and found a real problem.
    """
    try:
        config = client.config_get("save")
        aof_config = client.config_get("appendonly")
        info = client.info(section="persistence")
    except redis.RedisError as exc:
        raise PersistenceCheckError(
            f"could not query Redis persistence configuration: {exc}"
        ) from exc

    save_points = (config or {}).get("save", "")
    rdb_enabled = bool(save_points and save_points.strip())

    aof_setting = (aof_config or {}).get("appendonly", "no")
    aof_enabled = aof_setting == "yes"

    last_bgsave_status = info.get("rdb_last_bgsave_status")
    last_aof_rewrite_status = info.get("aof_last_bgrewrite_status") if aof_enabled else None

    issues: list[str] = []

    if not rdb_enabled and not aof_enabled:
        issues.append(
            "no persistence mechanism enabled (RDB save points empty and "
            "appendonly is off); this Redis instance loses its entire "
            "dataset on restart or failover"
        )

    if rdb_enabled and last_bgsave_status == "err":
        issues.append("last RDB background save failed (rdb_last_bgsave_status=err)")

    if aof_enabled and last_aof_rewrite_status == "err":
        issues.append("last AOF background rewrite failed (aof_last_bgrewrite_status=err)")

    return PersistenceHealth(
        healthy=not issues,
        rdb_enabled=rdb_enabled,
        aof_enabled=aof_enabled,
        last_bgsave_status=last_bgsave_status,
        last_aof_rewrite_status=last_aof_rewrite_status,
        issues=issues,
    )


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
