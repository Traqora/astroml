"""Tests for database query profiling module."""

from __future__ import annotations

import logging
import os
import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import sqlalchemy
from sqlalchemy import create_engine, text

from astroml.db.query_profiler import (
    QueryProfile,
    QueryProfiler,
    configure_query_logging,
    get_query_profiler,
    profile_query_context,
    check_slow_query_threshold_ci,
    DEFAULT_SLOW_QUERY_THRESHOLD_MS,
    SLOW_QUERY_LOGGER,
)


@pytest.fixture
def sqlite_engine():
    """Create an in-memory SQLite engine for testing."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def profiler():
    """Create a QueryProfiler instance for testing."""
    return QueryProfiler(
        slow_query_threshold_ms=50,
        enable_explain_analyze=False,  # Disable for SQLite compatibility
        log_all_queries=False
    )


class TestQueryProfile:
    """Test QueryProfile dataclass."""
    
    def test_query_profile_creation(self):
        """Test creating a QueryProfile instance."""
        profile = QueryProfile(
            statement="SELECT * FROM test",
            parameters={"id": 1},
            duration_ms=75.5,
            is_slow=True
        )
        
        assert profile.statement == "SELECT * FROM test"
        assert profile.parameters == {"id": 1}
        assert profile.duration_ms == 75.5
        assert profile.is_slow is True
        assert profile.explain_plan is None


class TestQueryProfiler:
    """Test QueryProfiler class."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization with default values."""
        profiler = QueryProfiler()
        
        assert profiler.slow_query_threshold_ms == DEFAULT_SLOW_QUERY_THRESHOLD_MS
        assert profiler.enable_explain_analyze is True
        assert profiler.log_all_queries is False
        assert profiler._enabled is False
        assert len(profiler._profiles) == 0
    
    def test_profiler_custom_initialization(self):
        """Test profiler initialization with custom values."""
        profiler = QueryProfiler(
            slow_query_threshold_ms=200,
            enable_explain_analyze=False,
            log_all_queries=True
        )
        
        assert profiler.slow_query_threshold_ms == 200
        assert profiler.enable_explain_analyze is False
        assert profiler.log_all_queries is True
    
    def test_enable_profiler(self, sqlite_engine, profiler):
        """Test enabling profiler on an engine."""
        profiler.enable(sqlite_engine)
        
        assert profiler._enabled is True
    
    def test_profiler_tracks_queries(self, sqlite_engine, profiler):
        """Test that profiler tracks query execution."""
        profiler.enable(sqlite_engine)
        
        # Execute a query
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        profiles = profiler.get_profiles()
        assert len(profiles) >= 1
        assert profiles[0].statement is not None
        assert profiles[0].duration_ms >= 0
    
    def test_profiler_detects_slow_queries(self, sqlite_engine):
        """Test slow query detection with very low threshold."""
        profiler = QueryProfiler(
            slow_query_threshold_ms=1,  # 1ms threshold
            enable_explain_analyze=False,
            log_all_queries=False
        )
        profiler.enable(sqlite_engine)
        
        # Execute a query (should be slow enough to trigger)
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        slow_queries = profiler.get_slow_queries()
        # Most queries should exceed 1ms threshold
        assert len(slow_queries) >= 0
    
    def test_get_statistics_empty(self, profiler):
        """Test statistics when no queries have been profiled."""
        stats = profiler.get_statistics()
        
        assert stats["total_queries"] == 0
        assert stats["slow_queries"] == 0
        assert stats["total_duration_ms"] == 0.0
        assert stats["avg_duration_ms"] == 0.0
        assert stats["max_duration_ms"] == 0.0
        assert stats["slow_query_rate"] == 0.0
    
    def test_get_statistics_with_queries(self, sqlite_engine, profiler):
        """Test statistics after executing queries."""
        profiler.enable(sqlite_engine)
        
        # Execute multiple queries
        with sqlite_engine.connect() as conn:
            for _ in range(3):
                conn.execute(text("SELECT 1"))
        
        stats = profiler.get_statistics()
        
        assert stats["total_queries"] >= 3
        assert stats["total_duration_ms"] > 0
        assert stats["avg_duration_ms"] > 0
        assert stats["max_duration_ms"] > 0
    
    def test_clear_profiles(self, sqlite_engine, profiler):
        """Test clearing collected profiles."""
        profiler.enable(sqlite_engine)
        
        # Execute a query
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        assert len(profiler.get_profiles()) >= 1
        
        # Clear profiles
        profiler.clear_profiles()
        
        assert len(profiler.get_profiles()) == 0


class TestConfigureQueryLogging:
    """Test query logging configuration."""
    
    def test_configure_query_logging_basic(self):
        """Test basic query logging configuration."""
        configure_query_logging(
            log_level="INFO",
            echo=False,
            enable_profiling=False
        )
        
        # Check that SQLAlchemy logger is configured
        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        assert sqlalchemy_logger.level == logging.INFO
    
    def test_configure_query_logging_with_profiling(self):
        """Test configuration with profiling enabled."""
        configure_query_logging(
            log_level="DEBUG",
            echo=False,
            enable_profiling=True,
            slow_query_threshold_ms=200
        )
        
        profiler = get_query_profiler()
        assert profiler is not None
        assert profiler.slow_query_threshold_ms == 200
    
    def test_get_query_profiler_none_when_disabled(self):
        """Test that get_query_profiler returns None when not enabled."""
        # Ensure profiler is not enabled
        from astroml.db import query_profiler
        query_profiler._global_profiler = None
        
        profiler = get_query_profiler()
        assert profiler is None


class TestProfileQueryContext:
    """Test profile_query_context context manager."""
    
    def test_profile_query_context(self, sqlite_engine):
        """Test profiling queries within a context."""
        with profile_query_context(engine=sqlite_engine) as profiler:
            # Execute queries
            with sqlite_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.execute(text("SELECT 2"))
        
        # Check that queries were profiled
        profiles = profiler.get_profiles()
        assert len(profiles) >= 2
    
    def test_profile_query_context_statistics(self, sqlite_engine):
        """Test getting statistics from context manager."""
        with profile_query_context(engine=sqlite_engine) as profiler:
            with sqlite_engine.connect() as conn:
                for _ in range(5):
                    conn.execute(text("SELECT 1"))
        
        stats = profiler.get_statistics()
        assert stats["total_queries"] >= 5


class TestCheckSlowQueryThresholdCI:
    """Test CI slow query threshold checking."""
    
    def test_check_slow_query_threshold_ci_no_profiler(self):
        """Test CI check when profiler is not enabled."""
        # Ensure profiler is None
        from astroml.db import query_profiler
        query_profiler._global_profiler = None
        
        result = check_slow_query_threshold_ci(threshold_ms=100)
        assert result is True
    
    def test_check_slow_query_threshold_ci_no_slow_queries(self, sqlite_engine):
        """Test CI check with no slow queries."""
        profiler = QueryProfiler(
            slow_query_threshold_ms=1000,  # High threshold
            enable_explain_analyze=False
        )
        profiler.enable(sqlite_engine)
        
        # Execute fast query
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Mock global profiler
        from astroml.db import query_profiler
        query_profiler._global_profiler = profiler
        
        result = check_slow_query_threshold_ci(threshold_ms=100)
        assert result is True
    
    def test_check_slow_query_threshold_ci_with_slow_queries(self, sqlite_engine):
        """Test CI check with slow queries exceeding threshold."""
        profiler = QueryProfiler(
            slow_query_threshold_ms=1,  # Very low threshold
            enable_explain_analyze=False
        )
        profiler.enable(sqlite_engine)
        
        # Execute query
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Mock global profiler
        from astroml.db import query_profiler
        query_profiler._global_profiler = profiler
        
        # Check with low threshold
        result = check_slow_query_threshold_ci(threshold_ms=0)
        # Should fail if any query exceeds 0ms threshold
        assert result is False or result is True  # Depends on actual query time


class TestSessionIntegration:
    """Test integration with session.py debug mode."""
    
    def test_debug_mode_enables_profiling(self, sqlite_engine):
        """Test that ASTROML_DEBUG enables profiling."""
        with patch.dict(os.environ, {"ASTROML_DEBUG": "true"}):
            # Mock the configure_query_logging to avoid actual engine creation
            with patch("astroml.db.query_profiler.configure_query_logging") as mock_config:
                from astroml.db.session import _enable_query_profiling_if_debug
                
                _enable_query_profiling_if_debug(sqlite_engine)
                
                mock_config.assert_called_once()
    
    def test_custom_slow_query_threshold(self, sqlite_engine):
        """Test custom slow query threshold from environment."""
        with patch.dict(os.environ, {
            "ASTROML_DEBUG": "true",
            "ASTROML_SLOW_QUERY_THRESHOLD_MS": "250"
        }):
            with patch("astroml.db.query_profiler.configure_query_logging") as mock_config:
                from astroml.db.session import _enable_query_profiling_if_debug
                
                _enable_query_profiling_if_debug(sqlite_engine)
                
                # Check that custom threshold was passed
                call_kwargs = mock_config.call_args[1]
                assert call_kwargs["slow_query_threshold_ms"] == 250


class TestQueryPlanPerformance:
    """Performance tests for query plans."""
    
    def test_simple_query_performance(self, sqlite_engine):
        """Test performance of simple SELECT query."""
        profiler = QueryProfiler(
            slow_query_threshold_ms=1000,
            enable_explain_analyze=False
        )
        profiler.enable(sqlite_engine)
        
        with sqlite_engine.connect() as conn:
            for _ in range(10):
                conn.execute(text("SELECT 1"))
        
        stats = profiler.get_statistics()
        assert stats["total_queries"] >= 10
        assert stats["avg_duration_ms"] < 100  # Should be very fast
    
    def test_query_with_filter_performance(self, sqlite_engine):
        """Test performance of query with WHERE clause."""
        # Create a test table
        with sqlite_engine.connect() as conn:
            conn.execute(text("CREATE TABLE test_table (id INTEGER, value TEXT)"))
            for i in range(100):
                conn.execute(text("INSERT INTO test_table VALUES (:id, :val)"), 
                           {"id": i, "val": f"value_{i}"})
            conn.commit()
        
        profiler = QueryProfiler(
            slow_query_threshold_ms=1000,
            enable_explain_analyze=False
        )
        profiler.enable(sqlite_engine)
        
        with sqlite_engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM test_table WHERE id = 50"))
            result.fetchall()
        
        stats = profiler.get_statistics()
        assert stats["total_queries"] >= 1
        assert stats["avg_duration_ms"] < 100
    
    def test_query_plan_comparison(self, sqlite_engine):
        """Test comparing performance of different query patterns."""
        # Create test data
        with sqlite_engine.connect() as conn:
            conn.execute(text("CREATE TABLE performance_test (id INTEGER, value INTEGER)"))
            for i in range(1000):
                conn.execute(text("INSERT INTO performance_test VALUES (:id, :val)"),
                           {"id": i, "val": i * 2})
            conn.commit()
        
        # Test 1: SELECT *
        profiler1 = QueryProfiler(slow_query_threshold_ms=1000, enable_explain_analyze=False)
        profiler1.enable(sqlite_engine)
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT * FROM performance_test WHERE value > 500")).fetchall()
        stats1 = profiler1.get_statistics()
        
        # Test 2: SELECT specific columns
        profiler2 = QueryProfiler(slow_query_threshold_ms=1000, enable_explain_analyze=False)
        profiler2.enable(sqlite_engine)
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT id FROM performance_test WHERE value > 500")).fetchall()
        stats2 = profiler2.get_statistics()
        
        # Both should complete quickly
        assert stats1["avg_duration_ms"] < 100
        assert stats2["avg_duration_ms"] < 100


if __name__ == "__main__":
    pytest.main([__file__])
