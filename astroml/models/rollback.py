"""Automated rollback to last-known-healthy version.

Provides a config-driven guard that rolls serving back to the
previous healthy version when health checks fail immediately after
activation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RollbackConfig:
    """Configuration for the automated rollback guard."""

    health_check_timeout: float = 30.0
    """Maximum seconds to wait for a health check after activation."""
    max_retries: int = 3
    """Number of health check retries before triggering rollback."""
    cooldown_seconds: float = 60.0
    """Minimum seconds between rollback attempts."""


class RollbackGuard:
    """Config-driven guard that rolls serving back to the previous
    healthy version when health checks fail immediately after activation.

    Tracks activation history and health check results. If a newly
    activated version fails health checks, the guard automatically
    rolls back to the last known healthy version.
    """

    def __init__(self, config: RollbackConfig | None = None) -> None:
        self.config = config or RollbackConfig()
        self.history: list[ActivationRecord] = []
        self.last_healthy_version: str | None = None
        self.last_rollback_time: float = 0.0

    def record_activation(self, version_id: str, timestamp: float | None = None) -> None:
        """Record that a version was activated."""
        timestamp = timestamp or time.time()
        record = ActivationRecord(
            version_id=version_id,
            activated_at=timestamp,
            health_status="pending",
        )
        self.history.append(record)

    def record_health_check(self, version_id: str, healthy: bool) -> str | None:
        """Record the result of a health check for an activated version.

        Returns the version_id to roll back to if the check failed and
        rollback is warranted, or None otherwise.
        """
        # Find the most recent pending record for this version.
        for record in reversed(self.history):
            if record.version_id == version_id and record.health_status == "pending":
                if healthy:
                    record.health_status = "healthy"
                    self.last_healthy_version = version_id
                else:
                    record.health_status = "unhealthy"
                    return self._determine_rollback(version_id)
                return None

        # No pending record found; record a new one.
        record = ActivationRecord(
            version_id=version_id,
            activated_at=time.time(),
            health_status="healthy" if healthy else "unhealthy",
        )
        self.history.append(record)
        if not healthy:
            return self._determine_rollback(version_id)
        return None

    def _determine_rollback(self, failed_version_id: str) -> str | None:
        """Determine if rollback is warranted and return the target version.

        Checks cooldown and finds the last known healthy version.
        """
        now = time.time()
        if now - self.last_rollback_time < self.config.cooldown_seconds:
            return None

        if self.last_healthy_version is None or self.last_healthy_version == failed_version_id:
            return None

        self.last_rollback_time = now
        return self.last_healthy_version

    def get_activation_history(self) -> list[ActivationRecord]:
        """Return the full activation history."""
        return list(self.history)

    def get_last_healthy(self) -> str | None:
        """Return the last known healthy version_id."""
        return self.last_healthy_version


@dataclass
class ActivationRecord:
    """Record of a single version activation and its health status."""

    version_id: str
    activated_at: float
    health_status: str  # "pending", "healthy", "unhealthy"
    rolled_back_at: float | None = None
    rolled_back_to: str | None = None

    def mark_rolled_back(self, rolled_back_to: str, timestamp: float | None = None) -> None:
        """Mark this activation as rolled back."""
        self.rolled_back_at = timestamp or time.time()
        self.rolled_back_to = rolled_back_to
