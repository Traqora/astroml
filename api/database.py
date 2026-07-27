"""Async database session management for the FastAPI backend (issue #251).

Provides:
  - Async SQLAlchemy engine + session factory
  - ``get_db`` FastAPI dependency (async)
  - ``get_sync_db`` for sync endpoints / scripts
  - Connection pool health checks and metrics (issue #297)
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

# Import all models so Base.metadata is fully populated before create_all.
from astroml.db.schema import Base  # noqa: F401
import api.models.orm  # noqa: F401  registers api models on Base.metadata


@dataclass
class PoolConfig:
    """Database connection pool configuration."""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    pool_pre_ping: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    interval: int = 60
    timeout: int = 5
    log_enabled: bool = True


class PoolMetrics:
    """Metrics for database connection pool monitoring."""

    def __init__(self):
        self._acquire_times: list[float] = []
        self._pool_usage: list[Dict[str, Any]] = []
        self._connection_errors: int = 0
        self._health_check_failures: int = 0

    def record_acquire_time(self, duration: float) -> None:
        """Record connection acquisition time."""
        self._acquire_times.append(duration)
        if len(self._acquire_times) > 1000:
            self._acquire_times = self._acquire_times[-1000:]

    def record_pool_usage(self, usage: Dict[str, Any]) -> None:
        """Record pool usage statistics."""
        self._pool_usage.append(usage)
        if len(self._pool_usage) > 100:
            self._pool_usage = self._pool_usage[-100:]

    def record_connection_error(self) -> None:
        """Record a connection error."""
        self._connection_errors += 1

    def record_health_check_failure(self) -> None:
        """Record a health check failure."""
        self._health_check_failures += 1

    def get_average_acquire_time(self) -> float:
        """Get average connection acquisition time in seconds."""
        if not self._acquire_times:
            return 0.0
        return sum(self._acquire_times) / len(self._acquire_times)

    def get_recent_pool_usage(self) -> list[Dict[str, Any]]:
        """Get recent pool usage statistics."""
        return self._pool_usage[-10:]

    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts."""
        return {
            "connection_errors": self._connection_errors,
            "health_check_failures": self._health_check_failures,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._acquire_times = []
        self._pool_usage = []
        self._connection_errors = 0
        self._health_check_failures = 0


def _async_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://astroml:astroml@localhost/astroml",
    )


def _sync_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://astroml:astroml@localhost/astroml",
    )
    return url.replace("+asyncpg", "").replace("+aiosqlite", "")


def _load_pool_config() -> PoolConfig:
    """Load pool configuration from database config."""
    try:
        from astroml.db.session import load_database_config
        config = load_database_config()
        
        # Try to get pool config from the loaded config
        pool_config = getattr(config, 'pool', None)
        if pool_config:
            return PoolConfig(
                min_size=getattr(pool_config, 'min_size', 5),
                max_size=getattr(pool_config, 'max_size', 20),
                max_overflow=getattr(pool_config, 'max_overflow', 10),
                pool_timeout=getattr(pool_config, 'pool_timeout', 30),
                pool_recycle=getattr(pool_config, 'pool_recycle', 1800),
                pool_pre_ping=getattr(pool_config, 'pool_pre_ping', True),
                max_retries=getattr(pool_config, 'max_retries', 3),
                retry_delay=getattr(pool_config, 'retry_delay', 1.0),
            )
    except Exception:
        pass
    
    # Fallback to environment variables
    return PoolConfig(
        min_size=int(os.environ.get("DB_POOL_MIN_SIZE", "5")),
        max_size=int(os.environ.get("DB_POOL_MAX_SIZE", "20")),
        max_overflow=int(os.environ.get("DB_POOL_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.environ.get("DB_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.environ.get("DB_POOL_RECYCLE", "1800")),
        pool_pre_ping=os.environ.get("DB_POOL_PRE_PING", "true").lower() == "true",
        max_retries=int(os.environ.get("DB_POOL_MAX_RETRIES", "3")),
        retry_delay=float(os.environ.get("DB_POOL_RETRY_DELAY", "1.0")),
    )


def _load_health_check_config() -> HealthCheckConfig:
    """Load health check configuration."""
    try:
        from astroml.db.session import load_database_config
        config = load_database_config()
        
        health_check = getattr(config, 'health_check', None)
        if health_check:
            return HealthCheckConfig(
                interval=getattr(health_check, 'interval', 60),
                timeout=getattr(health_check, 'timeout', 5),
                log_enabled=getattr(health_check, 'log_enabled', True),
            )
    except Exception:
        pass
    
    return HealthCheckConfig(
        interval=int(os.environ.get("DB_HEALTH_CHECK_INTERVAL", "60")),
        timeout=int(os.environ.get("DB_HEALTH_CHECK_TIMEOUT", "5")),
        log_enabled=os.environ.get("DB_HEALTH_CHECK_LOG", "true").lower() == "true",
    )


# Global metrics instance
_pool_metrics = PoolMetrics()


@lru_cache(maxsize=1)
def _async_engine():
    pool_config = _load_pool_config()
    health_config = _load_health_check_config()
    
    return create_async_engine(
        _async_url(),
        pool_pre_ping=pool_config.pool_pre_ping,
        pool_size=pool_config.max_size,
        max_overflow=pool_config.max_overflow,
        pool_timeout=pool_config.pool_timeout,
        pool_recycle=pool_config.pool_recycle,
        pool_use_lifo=True,  # LIFO reduces connection churn
        echo=False,
        echo_pool=False,
    )


@lru_cache(maxsize=1)
def _sync_engine():
    pool_config = _load_pool_config()
    
    return create_engine(
        _sync_url(),
        pool_pre_ping=pool_config.pool_pre_ping,
        pool_size=pool_config.max_size,
        max_overflow=pool_config.max_overflow,
        pool_timeout=pool_config.pool_timeout,
        pool_recycle=pool_config.pool_recycle,
        pool_use_lifo=True,
        echo=False,
        echo_pool=False,
    )


def get_async_engine():
    """Return the shared async engine (used by health probes, issue #550)."""
    return _async_engine()


def reset_engines() -> None:
    """Clear cached engines (used in tests when DATABASE_URL changes)."""
    _async_engine.cache_clear()
    _sync_engine.cache_clear()
    _pool_metrics.reset()


def _async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=_async_engine(),
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory (used by scheduler and WS)."""
    return _async_session_factory()


def _sync_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=_sync_engine(),
        autocommit=False,
        autoflush=False,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async DB session."""
    factory = _async_session_factory()
    start_time = time.time()
    
    try:
        async with factory() as session:
            acquire_time = time.time() - start_time
            _pool_metrics.record_acquire_time(acquire_time)
            
            # Record pool usage statistics
            pool = session.get_bind().pool
            if hasattr(pool, 'size'):
                _pool_metrics.record_pool_usage({
                    "size": pool.size(),
                    "checkedin": pool.checkedin(),
                    "overflow": pool.overflow(),
                    "total": pool.total(),
                })
            
            yield session
    except Exception as e:
        _pool_metrics.record_connection_error()
        raise e


def get_sync_db() -> Generator[Session, None, None]:
    """FastAPI dependency for sync endpoints — yields a sync DB session."""
    session = _sync_session_factory()()
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def get_db_with_retry(max_retries: int = 3, retry_delay: float = 1.0) -> AsyncGenerator[AsyncSession, None]:
    """Get a database session with retry logic for connection failures."""
    pool_config = _load_pool_config()
    max_retries = max_retries or pool_config.max_retries
    retry_delay = retry_delay or pool_config.retry_delay
    
    last_error = None
    for attempt in range(max_retries):
        try:
            async for session in get_db():
                yield session
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Log the error and retry
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    # All retries failed
    raise last_error or Exception("Failed to connect to database after retries")


async def check_database_connection() -> Dict[str, Any]:
    """
    Check database connection health and return status details.
    
    Returns:
        Dict with status, latency, pool stats, and error counts.
    """
    import logging
    logger = logging.getLogger(__name__)
    health_config = _load_health_check_config()
    
    result = {
        "status": "healthy",
        "latency_ms": 0.0,
        "pool_stats": {},
        "metrics": {},
        "error": None,
    }
    
    try:
        start_time = time.time()
        async with get_async_session_factory()() as session:
            # Execute a simple query to check connection
            await session.execute(text("SELECT 1"))
            latency_ms = (time.time() - start_time) * 1000
            result["latency_ms"] = round(latency_ms, 2)
            
            # Get pool statistics
            pool = session.get_bind().pool
            if hasattr(pool, 'size'):
                result["pool_stats"] = {
                    "size": pool.size(),
                    "checkedin": pool.checkedin(),
                    "overflow": pool.overflow(),
                    "total": pool.total(),
                    "max_size": pool.size() + pool.overflow(),
                }
            
            # Get metrics
            result["metrics"] = {
                "avg_acquire_time_ms": round(_pool_metrics.get_average_acquire_time() * 1000, 2),
                "connection_errors": _pool_metrics.get_error_counts()["connection_errors"],
                "health_check_failures": _pool_metrics.get_error_counts()["health_check_failures"],
            }
            
            if health_config.log_enabled:
                logger.info(f"Database health check passed (latency: {result['latency_ms']}ms)")
                
    except Exception as e:
        result["status"] = "unhealthy"
        result["error"] = str(e)
        _pool_metrics.record_health_check_failure()
        if health_config.log_enabled:
            logger.error(f"Database health check failed: {e}")
    
    return result


def get_pool_metrics() -> Dict[str, Any]:
    """Get connection pool metrics for monitoring."""
    return {
        "avg_acquire_time_ms": round(_pool_metrics.get_average_acquire_time() * 1000, 2),
        "recent_usage": _pool_metrics.get_recent_pool_usage(),
        "error_counts": _pool_metrics.get_error_counts(),
    }


async def get_db_status() -> Dict[str, Any]:
    """Get database status including health and metrics."""
    health = await check_database_connection()
    metrics = get_pool_metrics()
    
    return {
        **health,
        "metrics": metrics,
    }