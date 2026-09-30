"""Tests for automated rollback to last-known-healthy version."""

from __future__ import annotations

import time

import pytest

from astroml.models.rollback import ActivationRecord, RollbackConfig, RollbackGuard


class TestRollbackGuard:
    def test_rolls_back_when_health_fails(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.record_activation("v2")
        guard.last_healthy_version = "v1"

        rollback_target = guard.record_health_check("v2", healthy=False)
        assert rollback_target == "v1"

    def test_no_rollback_when_healthy(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.last_healthy_version = "v1"

        rollback_target = guard.record_health_check("v1", healthy=True)
        assert rollback_target is None

    def test_no_rollback_without_healthy_version(self):
        guard = RollbackGuard()
        guard.record_activation("v1")

        rollback_target = guard.record_health_check("v1", healthy=False)
        assert rollback_target is None

    def test_no_rollback_without_previous_version(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.record_activation("v2")
        # No last_healthy_version set yet
        rollback_target = guard.record_health_check("v2", healthy=False)
        assert rollback_target is None

    def test_cooldown_prevents_immediate_rollback(self):
        guard = RollbackGuard(config=RollbackConfig(cooldown_seconds=999))
        guard.record_activation("v1")
        guard.record_activation("v2")
        guard.last_healthy_version = "v1"
        guard.last_rollback_time = time.time()

        rollback_target = guard.record_health_check("v2", healthy=False)
        assert rollback_target is None

    def test_cooldown_allows_rollback_after_expiry(self):
        guard = RollbackGuard(config=RollbackConfig(cooldown_seconds=0))
        guard.record_activation("v1")
        guard.record_activation("v2")
        guard.last_healthy_version = "v1"
        guard.last_rollback_time = 0.0  # Expired

        rollback_target = guard.record_health_check("v2", healthy=False)
        assert rollback_target == "v1"

    def test_activation_history(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.record_activation("v2")

        history = guard.get_activation_history()
        assert len(history) == 2
        assert history[0].version_id == "v1"
        assert history[1].version_id == "v2"

    def test_health_check_preserves_last_healthy(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.record_health_check("v1", healthy=True)
        assert guard.get_last_healthy() == "v1"

    def test_record_activation_sets_pending(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        history = guard.get_activation_history()
        assert history[0].health_status == "pending"

    def test_record_health_check_marks_healthy(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.record_health_check("v1", healthy=True)
        history = guard.get_activation_history()
        assert history[0].health_status == "healthy"

    def test_record_health_check_marks_unhealthy(self):
        guard = RollbackGuard()
        guard.record_activation("v1")
        guard.record_health_check("v1", healthy=False)
        history = guard.get_activation_history()
        assert history[0].health_status == "unhealthy"

    def test_activation_record_mark_rolled_back(self):
        record = ActivationRecord(version_id="v1", activated_at=time.time())
        record.mark_rolled_back("v0")
        assert record.rolled_back_to == "v0"
        assert record.rolled_back_at is not None

    def test_default_config(self):
        guard = RollbackGuard()
        assert guard.config.health_check_timeout == 30.0
        assert guard.config.max_retries == 3
        assert guard.config.cooldown_seconds == 60.0

    def test_custom_config(self):
        config = RollbackConfig(health_check_timeout=10.0, max_retries=5, cooldown_seconds=120.0)
        guard = RollbackGuard(config=config)
        assert guard.config.health_check_timeout == 10.0
        assert guard.config.max_retries == 5
        assert guard.config.cooldown_seconds == 120.0


class TestRollbackGuardIntegration:
    def test_full_rollback_workflow(self):
        """Simulates a full activation -> health check failure -> rollback flow."""
        guard = RollbackGuard()

        # Activate v1 (baseline healthy)
        guard.record_activation("v1")
        result = guard.record_health_check("v1", healthy=True)
        assert guard.get_last_healthy() == "v1"
        assert result is None

        # Activate v2 (new version)
        guard.record_activation("v2")

        # v2 fails health checks → rollback to v1
        result = guard.record_health_check("v2", healthy=False)
        assert result == "v1"

        # v1 is still the last healthy version
        assert guard.get_last_healthy() == "v1"

    def test_multiple_activations_tracked(self):
        guard = RollbackGuard()
        for i in range(5):
            guard.record_activation(f"v{i}")

        assert len(guard.get_activation_history()) == 5

    def test_rollback_with_no_cooldown_and_healthy_baseline(self):
        guard = RollbackGuard(config=RollbackConfig(cooldown_seconds=0))
        guard.record_activation("v1")
        guard.record_health_check("v1", healthy=True)
        guard.record_activation("v2")

        result = guard.record_health_check("v2", healthy=False)
        assert result == "v1"
