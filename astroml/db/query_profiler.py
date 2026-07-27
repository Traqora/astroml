"""Database query profiling and slow query detection.

This module provides SQLAlchemy query profiling capabilities including:
- Query logging with configurable verbosity
- Slow query detection and EXPLAIN ANALYZE
- Custom logging for performance monitoring
- Integration with existing session management
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from sqlalchemy import event
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import Session
from sqlalchemy.sql import ClauseElement

logger = logging.getLogger(__name__)

# Default slow query threshold in milliseconds
DEFAULT_SLOW_QUERY_THRESHOLD_MS = 100

# Slow query logger
SLOW_QUERY_LOGGER = logging.getLogger("astroml.db.slow_queries")


@dataclass
class QueryProfile:
    """Profile information for a single query execution.
    
    Attributes:
        statement: SQL statement that was executed
        parameters: Query parameters
        duration_ms: Execution time in milliseconds
        is_slow: Whether query exceeded slow threshold
        explain_plan: EXPLAIN ANALYZE result (if slow)
    """
    
    statement: str
    parameters: Dict[str, Any]
    duration_ms: float
    is_slow: bool
    explain_plan: Optional[str] = None


class QueryProfiler:
    """Query profiler for SQLAlchemy engines.
    
    Tracks query execution times and automatically runs EXPLAIN ANALYZE
    on slow queries exceeding the configured threshold.
    """
    
    def __init__(
        self,
        slow_query_threshold_ms: int = DEFAULT_SLOW_QUERY_THRESHOLD_MS,
        enable_explain_analyze: bool = True,
        log_all_queries: bool = False
    ):
        """Initialize query profiler.
        
        Args:
            slow_query_threshold_ms: Threshold in ms for slow query detection
            enable_explain_analyze: Whether to run EXPLAIN ANALYZE on slow queries
            log_all_queries: Whether to log all queries (not just slow ones)
        """
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.enable_explain_analyze = enable_explain_analyze
        self.log_all_queries = log_all_queries
        self._profiles: list[QueryProfile] = []
        self._enabled = False
    
    def enable(self, engine: Engine) -> None:
        """Enable query profiling on the given engine.
        
        Args:
            engine: SQLAlchemy engine to profile
        """
        if self._enabled:
            return
        
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(
            conn: Connection,
            cursor: Any,
            statement: str,
            parameters: Dict[str, Any],
            context: Any,
            executemany: bool
        ) -> None:
            context._query_start_time = time.perf_counter()
        
        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(
            conn: Connection,
            cursor: Any,
            statement: str,
            parameters: Dict[str, Any],
            context: Any,
            executemany: bool
        ) -> None:
            if not hasattr(context, "_query_start_time"):
                return
            
            duration_ms = (time.perf_counter() - context._query_start_time) * 1000
            is_slow = duration_ms > self.slow_query_threshold_ms
            
            profile = QueryProfile(
                statement=statement,
                parameters=parameters,
                duration_ms=duration_ms,
                is_slow=is_slow
            )
            
            self._profiles.append(profile)
            
            if self.log_all_queries:
                logger.debug(
                    f"Query executed in {duration_ms:.2f}ms: {statement[:100]}..."
                )
            
            if is_slow:
                self._handle_slow_query(conn, statement, parameters, profile)
        
        self._enabled = True
        logger.info(f"Query profiling enabled (threshold: {self.slow_query_threshold_ms}ms)")
    
    def _handle_slow_query(
        self,
        conn: Connection,
        statement: str,
        parameters: Dict[str, Any],
        profile: QueryProfile
    ) -> None:
        """Handle a slow query by logging and optionally running EXPLAIN ANALYZE.
        
        Args:
            conn: Database connection
            statement: SQL statement
            parameters: Query parameters
            profile: Query profile to update
        """
        SLOW_QUERY_LOGGER.warning(
            f"Slow query detected ({profile.duration_ms:.2f}ms > {self.slow_query_threshold_ms}ms): "
            f"{statement[:200]}..."
        )
        
        if self.enable_explain_analyze:
            try:
                explain_result = self._run_explain_analyze(conn, statement, parameters)
                profile.explain_plan = explain_result
                
                SLOW_QUERY_LOGGER.warning(
                    f"EXPLAIN ANALYZE for slow query:\n{explain_result}"
                )
            except Exception as e:
                SLOW_QUERY_LOGGER.error(
                    f"Failed to run EXPLAIN ANALYZE: {e}"
                )
    
    def _run_explain_analyze(
        self,
        conn: Connection,
        statement: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Run EXPLAIN ANALYZE on a query.
        
        Args:
            conn: Database connection
            statement: SQL statement to explain
            parameters: Query parameters
            
        Returns:
            EXPLAIN ANALYZE output as string
        """
        explain_statement = f"EXPLAIN ANALYZE {statement}"
        
        try:
            result = conn.execute(explain_statement, parameters)
            rows = result.fetchall()
            return "\n".join(str(row[0]) for row in rows)
        except Exception as e:
            # Fallback to EXPLAIN if ANALYZE fails
            logger.debug(f"EXPLAIN ANALYZE failed, trying EXPLAIN: {e}")
            explain_statement = f"EXPLAIN {statement}"
            result = conn.execute(explain_statement, parameters)
            rows = result.fetchall()
            return "\n".join(str(row[0]) for row in rows)
    
    def get_profiles(self) -> list[QueryProfile]:
        """Get all collected query profiles.
        
        Returns:
            List of query profiles
        """
        return self._profiles.copy()
    
    def get_slow_queries(self) -> list[QueryProfile]:
        """Get only slow query profiles.
        
        Returns:
            List of slow query profiles
        """
        return [p for p in self._profiles if p.is_slow]
    
    def clear_profiles(self) -> None:
        """Clear all collected profiles."""
        self._profiles.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics.
        
        Returns:
            Dictionary with profiling statistics
        """
        if not self._profiles:
            return {
                "total_queries": 0,
                "slow_queries": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "slow_query_rate": 0.0
            }
        
        durations = [p.duration_ms for p in self._profiles]
        slow_count = len(self.get_slow_queries())
        
        return {
            "total_queries": len(self._profiles),
            "slow_queries": slow_count,
            "total_duration_ms": sum(durations),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "slow_query_rate": slow_count / len(self._profiles)
        }


# Global profiler instance
_global_profiler: Optional[QueryProfiler] = None


def configure_query_logging(
    log_level: str = "INFO",
    echo: bool = False,
    enable_profiling: bool = False,
    slow_query_threshold_ms: int = DEFAULT_SLOW_QUERY_THRESHOLD_MS
) -> None:
    """Configure SQLAlchemy query logging and profiling.
    
    Args:
        log_level: Logging level for SQLAlchemy engine logger
        echo: Whether to enable SQLAlchemy echo (prints all queries)
        enable_profiling: Whether to enable query profiling
        slow_query_threshold_ms: Threshold for slow query detection
    """
    # Configure SQLAlchemy engine logger
    sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
    sqlalchemy_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Enable query profiling if requested
    if enable_profiling:
        global _global_profiler
        _global_profiler = QueryProfiler(
            slow_query_threshold_ms=slow_query_threshold_ms,
            enable_explain_analyze=True,
            log_all_queries=(log_level.upper() == "DEBUG")
        )
        
        # Attach to existing engine if available
        try:
            from astroml.db.session import get_engine
            _global_profiler.enable(get_engine())
        except Exception as e:
            logger.warning(f"Could not enable profiling on existing engine: {e}")


def get_query_profiler() -> Optional[QueryProfiler]:
    """Get the global query profiler instance.
    
    Returns:
        QueryProfiler instance if enabled, None otherwise
    """
    return _global_profiler


@contextmanager
def profile_query_context(engine: Optional[Engine] = None):
    """Context manager for profiling a block of queries.
    
    Args:
        engine: Optional SQLAlchemy engine to profile. If None, uses get_engine().
        
    Yields:
        QueryProfiler instance for the context
        
    Example:
        with profile_query_context() as profiler:
            # Run queries
            session.query(Account).all()
            
        # Get statistics
        stats = profiler.get_statistics()
    """
    profiler = QueryProfiler(
        slow_query_threshold_ms=DEFAULT_SLOW_QUERY_THRESHOLD_MS,
        enable_explain_analyze=True
    )
    
    if engine is None:
        try:
            from astroml.db.session import get_engine
            engine = get_engine()
        except Exception as e:
            logger.warning(f"Could not get engine: {e}")
            yield profiler
            return
    
    try:
        profiler.enable(engine)
    except Exception as e:
        logger.warning(f"Could not enable profiler: {e}")
    
    try:
        yield profiler
    finally:
        # Clean up event listeners
        try:
            event.remove(engine, "before_cursor_execute")
            event.remove(engine, "after_cursor_execute")
        except Exception:
            pass


def check_slow_query_threshold_ci(threshold_ms: int = 100) -> bool:
    """Check if slow queries exceed threshold for CI/CD.
    
    This function is intended for use in CI pipelines to fail builds
    if queries are slower than expected.
    
    Args:
        threshold_ms: Maximum allowed slow query count threshold
        
    Returns:
        True if slow queries are within acceptable limits, False otherwise
    """
    profiler = get_query_profiler()
    if profiler is None:
        return True
    
    slow_queries = profiler.get_slow_queries()
    
    if not slow_queries:
        return True
    
    # Check if any query exceeds threshold
    for query in slow_queries:
        if query.duration_ms > threshold_ms:
            logger.error(
                f"CI check failed: Query exceeded {threshold_ms}ms threshold "
                f"({query.duration_ms:.2f}ms): {query.statement[:100]}..."
            )
            if query.explain_plan:
                logger.error(f"Query plan:\n{query.explain_plan}")
            return False
    
    return True
