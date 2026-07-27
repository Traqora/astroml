"""Granular health check endpoints (Issue #569) and pool metrics (#550).

Endpoints
---------

===========================  ==================================================
``GET /healthz``             Aggregate status across every dependency.
``GET /healthz/live``        Liveness — the process is running.
``GET /healthz/startup``     Startup probe — initialisation finished.
``GET /healthz/ready``       Readiness gate — safe to route traffic here.
``GET /healthz/db``          Database connectivity plus pool saturation.
``GET /healthz/cache``       Redis connectivity.
``GET /healthz/disk``        Free disk space on the data volume.
``GET /metrics/db-pool``     Connection pool utilization snapshot.
===========================  ==================================================

Every response uses the same JSON envelope::

    {"status": "ok|degraded|fail", "details": {...}, "remediation": "..."}

``fail`` is served with HTTP 503 so Kubernetes removes the pod from the
Service endpoints; ``degraded`` stays on 200 because the instance can still
serve traffic.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Awaitable, Callable, Final

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from api.database import get_async_engine, get_async_session_factory
from astroml.db.pool_health import check_pool, collect_pool_stats
from astroml.observability.health import (
    HTTP_STATUS_FOR,
    CheckResult,
    HealthStatus,
    aggregate_status,
    check_disk,
    readiness_state,
)
from astroml.observability.metrics import update_db_pool_metrics

router = APIRouter(tags=["health"])

#: Per-dependency probe timeout, seconds.
CHECK_TIMEOUT_SECONDS: Final[float] = float(
    os.environ.get("HEALTHZ_TIMEOUT_SECONDS", "5")
)

#: Path whose filesystem the disk probe inspects.
DISK_PATH: Final[str] = os.environ.get("HEALTHZ_DISK_PATH", ".")

#: When true, an unreachable cache fails readiness instead of degrading it.
CACHE_REQUIRED: Final[bool] = os.environ.get(
    "HEALTHZ_CACHE_REQUIRED", "false"
).lower() in ("1", "true", "yes")


def _envelope(result: CheckResult) -> JSONResponse:
    """Serve a single check result with the matching HTTP status."""
    return JSONResponse(status_code=result.http_status, content=result.to_dict())


def _aggregate_envelope(results: list[CheckResult], *, probe: str) -> JSONResponse:
    """Serve several check results as one aggregate response."""
    status = aggregate_status(results)
    remediation = " ".join(r.remediation for r in results if r.remediation)
    body: dict[str, Any] = {
        "status": status.value,
        "probe": probe,
        "details": {r.name: r.to_dict() for r in results},
        "remediation": remediation,
    }
    return JSONResponse(status_code=HTTP_STATUS_FOR[status], content=body)


async def _with_timeout(
    name: str,
    coro_factory: Callable[[], Awaitable[CheckResult]],
    timeout: float = CHECK_TIMEOUT_SECONDS,
) -> CheckResult:
    """Run a check, converting timeouts and crashes into ``FAIL`` results."""
    started = time.perf_counter()
    try:
        return await asyncio.wait_for(coro_factory(), timeout=timeout)
    except asyncio.TimeoutError:
        return CheckResult(
            name=name,
            status=HealthStatus.FAIL,
            details={"error": f"timed out after {timeout}s"},
            remediation=(
                f"The {name} check exceeded {timeout}s. Check network "
                "reachability and whether the dependency is saturated."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # noqa: BLE001 - a probe must always answer
        return CheckResult(
            name=name,
            status=HealthStatus.FAIL,
            details={"error_type": type(exc).__name__},
            remediation=f"The {name} check raised an unexpected error: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000,
        )


# ─── Individual checks ───────────────────────────────────────────────────────


async def check_database() -> CheckResult:
    """Verify database connectivity and fold in pool saturation.

    Returns:
        A ``CheckResult`` named ``"db"``. ``FAIL`` when ``SELECT 1`` does not
        succeed or the pool is exhausted; ``DEGRADED`` when the pool is above
        the 80% utilization alert threshold.
    """
    started = time.perf_counter()
    try:
        async with get_async_session_factory()() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return CheckResult(
            name="db",
            status=HealthStatus.FAIL,
            details={"error_type": type(exc).__name__},
            remediation=(
                "The database is unreachable. Verify DATABASE_URL, that the "
                "server accepts connections, and that credentials and network "
                "policy allow this pod to connect."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    latency_ms = (time.perf_counter() - started) * 1000
    pool_result = _pool_check()
    details: dict[str, Any] = {
        "latency_ms": round(latency_ms, 2),
        "query": "SELECT 1",
        "pool": dict(pool_result.details),
    }

    return CheckResult(
        name="db",
        status=pool_result.status,
        details=details,
        remediation=pool_result.remediation,
        duration_ms=latency_ms,
    )


def _pool_check() -> CheckResult:
    """Inspect the async engine's pool and refresh the Prometheus gauges."""
    engine = get_async_engine()
    stats = collect_pool_stats(engine)
    update_db_pool_metrics(stats)
    return check_pool(engine)


def _ping_redis(url: str) -> float:
    """Ping Redis synchronously, returning the round-trip time in ms."""
    import redis  # noqa: PLC0415 - optional dependency, imported lazily

    client = redis.Redis.from_url(
        url,
        socket_connect_timeout=CHECK_TIMEOUT_SECONDS,
        socket_timeout=CHECK_TIMEOUT_SECONDS,
    )
    try:
        started = time.perf_counter()
        client.ping()
        return (time.perf_counter() - started) * 1000
    finally:
        client.close()


async def check_cache() -> CheckResult:
    """Verify Redis connectivity.

    A cache outage is reported as ``DEGRADED`` (the API serves traffic
    without a cache) unless ``HEALTHZ_CACHE_REQUIRED`` is set, in which case
    it is ``FAIL`` and readiness drops.

    Returns:
        A ``CheckResult`` named ``"cache"``.
    """
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    started = time.perf_counter()
    failure_status = HealthStatus.FAIL if CACHE_REQUIRED else HealthStatus.DEGRADED

    try:
        latency_ms = await asyncio.to_thread(_ping_redis, url)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return CheckResult(
            name="cache",
            status=failure_status,
            details={
                "error_type": type(exc).__name__,
                "required": CACHE_REQUIRED,
            },
            remediation=(
                "Redis is unreachable. Verify REDIS_URL, that the Redis "
                "service is running, and that it is not evicting under "
                "memory pressure. Cached reads will fall through to the "
                "database until this recovers."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    return CheckResult(
        name="cache",
        status=HealthStatus.OK,
        details={"latency_ms": round(latency_ms, 2), "required": CACHE_REQUIRED},
        duration_ms=(time.perf_counter() - started) * 1000,
    )


async def check_disk_space() -> CheckResult:
    """Check free space on the configured data volume."""
    return await asyncio.to_thread(check_disk, DISK_PATH)


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.get("/healthz", summary="Aggregate health across all dependencies")
async def healthz() -> JSONResponse:
    """Return the combined status of every dependency."""
    results = await asyncio.gather(
        _with_timeout("db", check_database),
        _with_timeout("cache", check_cache),
        _with_timeout("disk", check_disk_space),
    )
    return _aggregate_envelope([readiness_state.snapshot(), *results], probe="healthz")


@router.get("/healthz/live", summary="Liveness probe")
async def healthz_live() -> JSONResponse:
    """Return 200 whenever the process can serve a request.

    Deliberately checks nothing external: a failing dependency must not get
    the container restarted.
    """
    return _envelope(
        CheckResult(
            name="live",
            status=HealthStatus.OK,
            details={"uptime_seconds": round(readiness_state.uptime_seconds, 2)},
        )
    )


@router.get("/healthz/startup", summary="Startup probe")
async def healthz_startup() -> JSONResponse:
    """Return 200 once application startup has completed."""
    return _envelope(readiness_state.snapshot())


@router.get("/healthz/ready", summary="Readiness probe")
async def healthz_ready() -> JSONResponse:
    """Gate traffic on startup completion and hard dependencies.

    The database is a hard dependency; the cache and disk checks can only
    downgrade readiness to ``degraded`` (still 200) unless they fail
    outright.
    """
    startup = readiness_state.snapshot()
    if startup.status is HealthStatus.FAIL:
        return _aggregate_envelope([startup], probe="ready")

    results = await asyncio.gather(
        _with_timeout("db", check_database),
        _with_timeout("cache", check_cache),
        _with_timeout("disk", check_disk_space),
    )
    return _aggregate_envelope([startup, *results], probe="ready")


@router.get("/healthz/db", summary="Database connectivity probe")
async def healthz_db() -> JSONResponse:
    """Check database connectivity and connection pool saturation."""
    return _envelope(await _with_timeout("db", check_database))


@router.get("/healthz/cache", summary="Redis connectivity probe")
async def healthz_cache() -> JSONResponse:
    """Check Redis connectivity."""
    return _envelope(await _with_timeout("cache", check_cache))


@router.get("/healthz/disk", summary="Disk space probe")
async def healthz_disk() -> JSONResponse:
    """Check free space on the data volume."""
    return _envelope(await _with_timeout("disk", check_disk_space))


@router.get("/metrics/db-pool", summary="Connection pool utilization")
async def db_pool_metrics() -> JSONResponse:
    """Return the connection pool snapshot as JSON.

    The same numbers are exported to Prometheus as ``db_pool_size`` and
    ``db_pool_utilization_ratio``; this endpoint exists for quick operator
    inspection without a Prometheus round trip.
    """
    result = _pool_check()
    body = {
        "status": result.status.value,
        "pool": dict(result.details),
        "remediation": result.remediation,
    }
    return JSONResponse(status_code=HTTP_STATUS_FOR[result.status], content=body)
