"""Redis persistence health check (issue #956).

``RedisCache`` treats Redis as a durable-enough store for feature/prediction
caches (``astroml/cache/redis_cache.py``), but nothing in this codebase
verifies that the connected Redis instance actually has persistence
(RDB snapshotting or AOF) enabled. A Redis instance running with both
disabled loses its entire dataset on restart or failover, silently
degrading every cache consumer to a 100% miss rate with no error raised
anywhere, since a cache miss is indistinguishable from an empty cache.

This module inspects Redis's own ``CONFIG GET``/``INFO persistence``
output (no local disk access required, works against a remote or
managed Redis instance) and reports whether at least one persistence
mechanism is active, plus recent save/rewrite failures if the server
reports any.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import redis

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
