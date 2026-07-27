"""Tests for health check primitives (Issue #569)."""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from astroml.observability.health import (
    HTTP_STATUS_FOR,
    CheckResult,
    HealthStatus,
    ReadinessState,
    aggregate_status,
    aggregate_to_dict,
    check_disk,
)


class _Usage:
    """Stand-in for ``shutil.disk_usage`` results."""

    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free
        self.used = total - free


class TestHealthStatus:
    @pytest.mark.parametrize(
        ("status", "serving"),
        [
            (HealthStatus.OK, True),
            (HealthStatus.DEGRADED, True),
            (HealthStatus.FAIL, False),
        ],
    )
    def test_is_serving(self, status: HealthStatus, serving: bool) -> None:
        assert status.is_serving is serving

    @pytest.mark.parametrize(
        ("status", "code"),
        [
            (HealthStatus.OK, 200),
            (HealthStatus.DEGRADED, 200),
            (HealthStatus.FAIL, 503),
        ],
    )
    def test_http_status_mapping(self, status: HealthStatus, code: int) -> None:
        assert HTTP_STATUS_FOR[status] == code


class TestCheckResult:
    def test_to_dict_shape(self) -> None:
        result = CheckResult(
            name="db",
            status=HealthStatus.DEGRADED,
            details={"latency_ms": 12.0},
            remediation="Scale the pool.",
            duration_ms=12.3456,
        )

        assert result.to_dict() == {
            "status": "degraded",
            "component": "db",
            "details": {"latency_ms": 12.0},
            "remediation": "Scale the pool.",
            "duration_ms": 12.35,
        }

    def test_http_status_for_failure(self) -> None:
        assert CheckResult("db", HealthStatus.FAIL).http_status == 503

    def test_defaults(self) -> None:
        result = CheckResult("live", HealthStatus.OK)
        assert result.details == {}
        assert result.remediation == ""


class TestAggregation:
    def test_empty_is_ok(self) -> None:
        assert aggregate_status([]) is HealthStatus.OK

    def test_worst_status_wins(self) -> None:
        results = [
            CheckResult("a", HealthStatus.OK),
            CheckResult("b", HealthStatus.DEGRADED),
            CheckResult("c", HealthStatus.FAIL),
        ]
        assert aggregate_status(results) is HealthStatus.FAIL

    def test_degraded_beats_ok(self) -> None:
        results = [
            CheckResult("a", HealthStatus.OK),
            CheckResult("b", HealthStatus.DEGRADED),
        ]
        assert aggregate_status(results) is HealthStatus.DEGRADED

    def test_aggregate_to_dict_merges_remediation(self) -> None:
        body = aggregate_to_dict(
            [
                CheckResult("a", HealthStatus.OK),
                CheckResult("b", HealthStatus.FAIL, remediation="Restart b."),
            ],
            remediation_prefix="Service is not ready.",
        )

        assert body["status"] == "fail"
        assert set(body["details"]) == {"a", "b"}
        assert body["remediation"] == "Service is not ready. Restart b."


class TestCheckDisk:
    def test_healthy_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(100, 60))

        result = check_disk("/data")

        assert result.status is HealthStatus.OK
        assert result.remediation == ""
        assert result.details["free_ratio"] == 0.6
        assert result.details["path"] == "/data"

    def test_degraded_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(100, 15))

        result = check_disk("/data")

        assert result.status is HealthStatus.DEGRADED
        assert "Schedule cleanup" in result.remediation

    def test_failing_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(100, 5))

        result = check_disk("/data")

        assert result.status is HealthStatus.FAIL
        assert result.http_status == 503
        assert "Free space immediately" in result.remediation

    def test_custom_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(100, 50))

        result = check_disk("/data", degraded_ratio=0.9, fail_ratio=0.6)

        assert result.status is HealthStatus.FAIL

    def test_zero_capacity_is_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(0, 0))

        assert check_disk("/data").status is HealthStatus.FAIL

    def test_oserror_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_path: Any) -> Any:
            raise OSError("not mounted")

        monkeypatch.setattr(shutil, "disk_usage", _boom)

        result = check_disk("/data")

        assert result.status is HealthStatus.FAIL
        assert result.details["error"] == "not mounted"
        assert "volume is mounted" in result.remediation


class TestReadinessState:
    def test_starts_not_ready(self) -> None:
        state = ReadinessState()

        assert state.started is False
        assert state.ready is False
        assert state.uptime_seconds == 0.0

        snapshot = state.snapshot()
        assert snapshot.status is HealthStatus.FAIL
        assert snapshot.details == {"started": False, "ready": False}
        assert "still initialising" in snapshot.remediation

    def test_ready_after_startup(self) -> None:
        state = ReadinessState()
        state.mark_started()

        assert state.started is True
        assert state.ready is True
        assert state.uptime_seconds >= 0.0
        assert state.snapshot().status is HealthStatus.OK

    def test_draining_revokes_readiness_but_keeps_started(self) -> None:
        state = ReadinessState()
        state.mark_started()
        state.set_ready(False, "Application is shutting down.")

        assert state.started is True
        assert state.ready is False

        snapshot = state.snapshot()
        assert snapshot.status is HealthStatus.FAIL
        assert "shutting down" in snapshot.remediation
        assert snapshot.details["started"] is True

    def test_set_ready_true_restores_default_detail(self) -> None:
        state = ReadinessState()
        state.mark_started()
        state.set_ready(False)
        state.set_ready(True)

        assert state.ready is True

    def test_reset_returns_to_initial_state(self) -> None:
        state = ReadinessState()
        state.mark_started()
        state.reset()

        assert state.started is False
        assert state.ready is False
