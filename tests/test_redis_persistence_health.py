"""Tests for Redis persistence health inspection (Issue #963)."""

from __future__ import annotations

from typing import Any

from astroml.cache.persistence_health import (
    PersistenceStats,
    check_persistence,
    collect_persistence_stats,
    evaluate_persistence_health,
)
from astroml.observability.health import HealthStatus


class _FakeRedisClient:
    """Minimal stand-in for ``redis.Redis`` exposing ``info``/``config_get``."""

    def __init__(self, info: dict[str, Any], save_value: str) -> None:
        self._info = info
        self._save_value = save_value

    def info(self, section: str | None = None) -> dict[str, Any]:
        return self._info

    def config_get(self, pattern: str) -> dict[str, str]:
        return {"save": self._save_value}


def _stats(**overrides: Any) -> PersistenceStats:
    values: dict[str, Any] = {
        "rdb_enabled": True,
        "aof_enabled": False,
        "rdb_last_bgsave_status": "ok",
        "aof_last_write_status": "unknown",
        "rdb_changes_since_last_save": 0,
    }
    values.update(overrides)
    return PersistenceStats(**values)


class TestPersistenceStats:
    def test_any_persistence_enabled_true_when_rdb_enabled(self) -> None:
        assert _stats(rdb_enabled=True, aof_enabled=False).any_persistence_enabled is True

    def test_any_persistence_enabled_true_when_aof_enabled(self) -> None:
        assert _stats(rdb_enabled=False, aof_enabled=True).any_persistence_enabled is True

    def test_any_persistence_enabled_false_when_neither(self) -> None:
        assert _stats(rdb_enabled=False, aof_enabled=False).any_persistence_enabled is False

    def test_to_dict_contains_all_fields(self) -> None:
        d = _stats().to_dict()
        assert set(d) == {
            "rdb_enabled",
            "aof_enabled",
            "rdb_last_bgsave_status",
            "aof_last_write_status",
            "rdb_changes_since_last_save",
        }


class TestCollectPersistenceStats:
    def test_rdb_enabled_when_save_configured(self) -> None:
        client = _FakeRedisClient(info={}, save_value="3600 1 300 100")
        stats = collect_persistence_stats(client)
        assert stats.rdb_enabled is True

    def test_rdb_disabled_when_save_empty(self) -> None:
        client = _FakeRedisClient(info={}, save_value="")
        stats = collect_persistence_stats(client)
        assert stats.rdb_enabled is False

    def test_aof_enabled_parsed_from_info(self) -> None:
        client = _FakeRedisClient(info={"aof_enabled": 1}, save_value="")
        stats = collect_persistence_stats(client)
        assert stats.aof_enabled is True

    def test_aof_disabled_parsed_from_info(self) -> None:
        client = _FakeRedisClient(info={"aof_enabled": 0}, save_value="")
        stats = collect_persistence_stats(client)
        assert stats.aof_enabled is False

    def test_missing_info_fields_use_safe_defaults(self) -> None:
        client = _FakeRedisClient(info={}, save_value="")
        stats = collect_persistence_stats(client)
        assert stats.rdb_last_bgsave_status == "unknown"
        assert stats.aof_last_write_status == "unknown"
        assert stats.rdb_changes_since_last_save == 0

    def test_reads_bgsave_and_write_status(self) -> None:
        client = _FakeRedisClient(
            info={
                "aof_enabled": 1,
                "rdb_last_bgsave_status": "ok",
                "aof_last_write_status": "err",
                "rdb_changes_since_last_save": 42,
            },
            save_value="3600 1",
        )
        stats = collect_persistence_stats(client)
        assert stats.rdb_last_bgsave_status == "ok"
        assert stats.aof_last_write_status == "err"
        assert stats.rdb_changes_since_last_save == 42


class TestEvaluatePersistenceHealth:
    def test_fails_when_no_persistence_enabled(self) -> None:
        result = evaluate_persistence_health(_stats(rdb_enabled=False, aof_enabled=False))
        assert result.status == HealthStatus.FAIL
        assert result.name == "redis_persistence"
        assert result.remediation

    def test_degraded_when_aof_last_write_failed(self) -> None:
        result = evaluate_persistence_health(
            _stats(rdb_enabled=False, aof_enabled=True, aof_last_write_status="err")
        )
        assert result.status == HealthStatus.DEGRADED
        assert "AOF write failed" in result.remediation

    def test_degraded_when_rdb_bgsave_failed(self) -> None:
        result = evaluate_persistence_health(
            _stats(rdb_enabled=True, aof_enabled=False, rdb_last_bgsave_status="err")
        )
        assert result.status == HealthStatus.DEGRADED
        assert "RDB background save failed" in result.remediation

    def test_ok_when_rdb_enabled_and_healthy(self) -> None:
        result = evaluate_persistence_health(
            _stats(rdb_enabled=True, aof_enabled=False, rdb_last_bgsave_status="ok")
        )
        assert result.status == HealthStatus.OK
        assert result.remediation == ""

    def test_ok_when_aof_enabled_and_healthy(self) -> None:
        result = evaluate_persistence_health(
            _stats(rdb_enabled=False, aof_enabled=True, aof_last_write_status="ok")
        )
        assert result.status == HealthStatus.OK
        assert result.remediation == ""

    def test_details_reflect_input_stats(self) -> None:
        stats = _stats(rdb_changes_since_last_save=7)
        result = evaluate_persistence_health(stats)
        assert result.details["rdb_changes_since_last_save"] == 7


class TestCheckPersistence:
    def test_returns_named_result_with_duration(self) -> None:
        client = _FakeRedisClient(
            info={"aof_enabled": 1, "aof_last_write_status": "ok"}, save_value=""
        )
        result = check_persistence(client)
        assert result.name == "redis_persistence"
        assert result.status == HealthStatus.OK
        assert result.duration_ms >= 0

    def test_end_to_end_fail_when_persistence_unconfigured(self) -> None:
        client = _FakeRedisClient(info={"aof_enabled": 0}, save_value="")
        result = check_persistence(client)
        assert result.status == HealthStatus.FAIL
