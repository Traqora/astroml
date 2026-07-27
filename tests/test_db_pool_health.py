"""Tests for connection pool health inspection (Issue #550)."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine

from astroml.db.pool_health import (
    POOL_UTILIZATION_ALERT_THRESHOLD,
    PoolStats,
    check_pool,
    collect_pool_stats,
    evaluate_pool_health,
)
from astroml.observability.health import HealthStatus


class _FakePool:
    """Minimal stand-in for ``QueuePool``."""

    def __init__(
        self,
        size: int = 10,
        checked_in: int = 5,
        checked_out: int = 5,
        overflow: int = -5,
        max_overflow: int = 10,
    ) -> None:
        self._size = size
        self._checked_in = checked_in
        self._checked_out = checked_out
        self._overflow = overflow
        self._max_overflow = max_overflow

    def size(self) -> int:
        return self._size

    def checkedin(self) -> int:
        return self._checked_in

    def checkedout(self) -> int:
        return self._checked_out

    def overflow(self) -> int:
        return self._overflow


class _FakeEngine:
    def __init__(self, pool: Any) -> None:
        self.pool = pool


class _FakeAsyncEngine:
    def __init__(self, pool: Any) -> None:
        self.sync_engine = _FakeEngine(pool)


def _stats(**overrides: Any) -> PoolStats:
    values: dict[str, Any] = {
        "pool_size": 10,
        "checked_in": 8,
        "checked_out": 2,
        "overflow": 0,
        "max_overflow": 10,
        "implementation": "QueuePool",
    }
    values.update(overrides)
    return PoolStats(**values)


class TestPoolStats:
    def test_capacity_is_size_plus_overflow(self) -> None:
        assert _stats().capacity == 20

    def test_utilization_ratio(self) -> None:
        assert _stats(checked_out=5).utilization == pytest.approx(0.25)

    def test_utilization_is_zero_for_empty_capacity(self) -> None:
        assert _stats(pool_size=0, max_overflow=0).utilization == 0.0

    def test_utilization_is_capped_at_one(self) -> None:
        assert _stats(checked_out=999).utilization == 1.0

    def test_saturation_threshold(self) -> None:
        assert _stats(checked_out=16).saturated is True
        assert _stats(checked_out=16).utilization == pytest.approx(
            POOL_UTILIZATION_ALERT_THRESHOLD
        )
        assert _stats(checked_out=15).saturated is False

    def test_exhaustion(self) -> None:
        assert _stats(checked_out=20).exhausted is True
        assert _stats(checked_out=19).exhausted is False

    def test_to_dict_exposes_operator_fields(self) -> None:
        body = _stats(checked_out=16).to_dict()

        assert body["utilization_percent"] == 80.0
        assert body["alert_threshold_percent"] == 80.0
        assert body["saturated"] is True
        assert body["exhausted"] is False
        assert body["implementation"] == "QueuePool"


class TestCollectPoolStats:
    def test_reads_counters_from_engine(self) -> None:
        stats = collect_pool_stats(_FakeEngine(_FakePool()))

        assert stats.pool_size == 10
        assert stats.checked_in == 5
        assert stats.checked_out == 5
        assert stats.max_overflow == 10
        assert stats.implementation == "_FakePool"

    def test_negative_overflow_is_clamped(self) -> None:
        assert collect_pool_stats(_FakeEngine(_FakePool(overflow=-5))).overflow == 0

    def test_async_engine_is_unwrapped(self) -> None:
        stats = collect_pool_stats(_FakeAsyncEngine(_FakePool(size=7)))
        assert stats.pool_size == 7

    def test_pool_without_counters_reports_zeros(self) -> None:
        class _Bare:
            pass

        stats = collect_pool_stats(_FakeEngine(_Bare()))

        assert stats.pool_size == 0
        assert stats.capacity == 0
        assert stats.utilization == 0.0

    def test_raising_counter_is_tolerated(self) -> None:
        class _Raising(_FakePool):
            def checkedout(self) -> int:
                raise RuntimeError("pool detached")

        assert collect_pool_stats(_FakeEngine(_Raising())).checked_out == 0

    def test_non_numeric_counter_is_tolerated(self) -> None:
        class _Weird(_FakePool):
            def size(self) -> Any:
                return "not-a-number"

        assert collect_pool_stats(_FakeEngine(_Weird())).pool_size == 0

    def test_real_sqlite_engine(self) -> None:
        engine = create_engine("sqlite://")
        stats = collect_pool_stats(engine)

        assert stats.checked_out >= 0
        assert stats.implementation


class TestEvaluatePoolHealth:
    def test_healthy_pool(self) -> None:
        result = evaluate_pool_health(_stats(checked_out=2))

        assert result.status is HealthStatus.OK
        assert result.remediation == ""
        assert result.name == "db_pool"

    def test_saturated_pool_is_degraded(self) -> None:
        result = evaluate_pool_health(_stats(checked_out=17))

        assert result.status is HealthStatus.DEGRADED
        assert result.http_status == 200
        assert "alert" in result.remediation
        assert "DB_POOL_MAX_SIZE" in result.remediation

    def test_exhausted_pool_fails(self) -> None:
        result = evaluate_pool_health(_stats(checked_out=20))

        assert result.status is HealthStatus.FAIL
        assert result.http_status == 503
        assert "pool_timeout" in result.remediation

    def test_zero_capacity_pool_is_ok(self) -> None:
        result = evaluate_pool_health(
            _stats(pool_size=0, max_overflow=0, checked_out=0)
        )
        assert result.status is HealthStatus.OK


class TestCheckPool:
    def test_check_pool_sets_duration(self) -> None:
        result = check_pool(_FakeEngine(_FakePool()))

        assert result.name == "db_pool"
        assert result.duration_ms >= 0.0
        assert result.details["pool_size"] == 10

    def test_check_pool_flags_saturation(self) -> None:
        result = check_pool(_FakeEngine(_FakePool(checked_out=20, max_overflow=10)))
        assert result.status is HealthStatus.FAIL


class TestSessionHelpers:
    def test_session_module_exposes_pool_helpers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from astroml.db import session as session_module

        engine = _FakeEngine(_FakePool(checked_out=1))
        monkeypatch.setattr(session_module, "get_engine", lambda: engine)

        assert session_module.get_pool_stats().checked_out == 1
        assert session_module.check_connection_pool().status is HealthStatus.OK

    def test_engine_enables_pre_ping_and_recycling(self) -> None:
        import inspect

        from astroml.db import session as session_module

        source = inspect.getsource(session_module.get_engine)

        assert "pool_pre_ping=True" in source
        assert "pool_recycle" in source
