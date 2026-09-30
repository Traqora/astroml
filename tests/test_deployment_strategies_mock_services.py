"""Deployment strategy tests driven by mock model services (#633, step 8).

These compose the pieces a real rollout uses — traffic weights, health checks,
metrics capture, automated rollback — instead of exercising one manager method
at a time.  The serving endpoint is an in-process stub, so the rollout
behaviour is asserted without a cluster.
"""

from __future__ import annotations

import pytest

from astroml.deployment.blue_green import BGPhase, BlueGreenConfig, BlueGreenManager
from astroml.deployment.canary import CanaryConfig, CanaryManager, CanaryPhase
from astroml.deployment.rollback_manager import RollbackManager
from astroml.deployment.traffic_router import RouteTarget, TrafficRouter


class MockModelService:
    """Stand-in for a served model version.

    Args:
        version: Version string the service serves.
        error_rate: Fraction of requests that fail.
        latency_ms: Reported average latency.
        reachable: Whether the endpoint answers at all.
    """

    def __init__(
        self,
        version: str,
        error_rate: float = 0.0,
        latency_ms: float = 20.0,
        reachable: bool = True,
    ) -> None:
        self.version = version
        self.error_rate = error_rate
        self.latency_ms = latency_ms
        self.reachable = reachable
        self.requests = 0

    def health(self) -> dict[str, object]:
        """Return the payload a health probe would read from the endpoint.

        Both managers gate on this: :class:`CanaryManager` on the metrics,
        :class:`BlueGreenManager` on ``healthy``.

        Returns:
            ``healthy`` plus the traffic metrics the rollout gates on.

        Raises:
            ConnectionError: When the stub endpoint is unreachable.
        """
        if not self.reachable:
            raise ConnectionError(f"{self.version} endpoint is down")
        self.requests += 1
        return {
            "healthy": self.error_rate <= 0.0,
            "error_rate": self.error_rate,
            "latency_ms": self.latency_ms,
            "version": self.version,
        }


def _canary_config(**overrides: object) -> CanaryConfig:
    """Build a fast canary config with sensible test defaults."""
    defaults: dict[str, object] = {
        "initial_weight": 5.0,
        "increment_step": 25.0,
        "max_canary_weight": 50.0,
        "stabilization_seconds": 0.0,
        "failure_threshold": 0.05,
        "latency_threshold_ms": 500.0,
        "auto_promote": True,
        "auto_rollback": True,
    }
    defaults.update(overrides)
    return CanaryConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_ramps_to_promotion_with_a_healthy_mock_service() -> None:
    """A clean canary ramps in steps and reaches 100% traffic."""
    router = TrafficRouter(seed=1)
    rollback = RollbackManager()
    manager = CanaryManager(rollback_manager=rollback)
    canary = MockModelService("v2.0.0")

    dep = manager.start_canary("fraud-graph", "v2.0.0", "v1.9.0", _canary_config())
    rule = router.add_rule(
        "fraud-graph",
        [
            RouteTarget("stable", "v1.9.0", weight=0.95),
            RouteTarget("canary", "v2.0.0", weight=0.05),
        ],
    )

    for _ in range(6):
        dep = manager.step(dep.deployment_id, canary.health)
        router.update_weights(
            rule.rule_id,
            {"stable": (100 - dep.current_weight) / 100, "canary": dep.current_weight / 100},
        )

    assert dep.phase is CanaryPhase.PROMOTED
    assert dep.current_weight == 100.0
    assert [s.weight for s in dep.steps] == [5.0, 30.0, 50.0]
    assert all(s.healthy for s in dep.steps)
    assert rollback.list_rollbacks() == []


def test_canary_steps_record_metrics_for_monitoring() -> None:
    """Step metrics are what a monitoring dashboard would scrape — none may be lost."""
    manager = CanaryManager()
    service = MockModelService("v2.0.0", error_rate=0.01, latency_ms=140.0)

    dep = manager.start_canary("fraud-graph", "v2.0.0", "v1.9.0", _canary_config())
    dep = manager.step(dep.deployment_id, service.health)

    step = dep.steps[0]
    assert step.step_number == 1
    assert step.error_rate == pytest.approx(0.01)
    assert step.avg_latency_ms == pytest.approx(140.0, abs=0.001)
    assert step.duration_seconds >= 0.0
    assert step.timestamp
    assert step.healthy


def test_canary_rolls_back_when_the_mock_service_breaches_the_error_budget() -> None:
    """Automated rollback: an unhealthy canary never reaches stable traffic."""
    rollback = RollbackManager()
    manager = CanaryManager(rollback_manager=rollback)
    broken = MockModelService("v2.0.0", error_rate=0.4)

    dep = manager.start_canary("fraud-graph", "v2.0.0", "v1.9.0", _canary_config())
    dep = manager.step(dep.deployment_id, broken.health)

    assert dep.phase is CanaryPhase.ROLLED_BACK
    assert dep.current_weight == 0.0

    records = rollback.list_rollbacks()
    assert len(records) == 1
    record = records[0]
    assert record.trigger.severity == "critical"
    assert record.status == "approved"
    assert record.trigger.target_version == "v1.9.0"
    assert record.trigger.context["model_name"] == "fraud-graph"


def test_rolled_back_canary_stops_receiving_routed_traffic() -> None:
    """A router synced to the canary weight must go quiet after a rollback."""
    router = TrafficRouter(seed=3)
    manager = CanaryManager()
    broken = MockModelService("v2.0.0", error_rate=0.4)

    dep = manager.start_canary("fraud-graph", "v2.0.0", "v1.9.0", _canary_config())
    rule = router.add_rule(
        "fraud-graph",
        [
            RouteTarget("stable", "v1.9.0", weight=0.95),
            RouteTarget("canary", "v2.0.0", weight=0.05),
        ],
    )
    dep = manager.step(dep.deployment_id, broken.health)
    router.update_weights(
        rule.rule_id,
        {"stable": (100 - dep.current_weight) / 100, "canary": dep.current_weight / 100},
    )

    routed: set[str] = set()
    for _ in range(200):
        target = router.route(rule.rule_id, strategy="weighted")
        assert target is not None
        routed.add(target.name)

    assert routed == {"stable"}


def test_canary_rolls_back_when_the_mock_service_is_too_slow() -> None:
    """Latency breaches roll back too, and are logged as high severity."""
    rollback = RollbackManager()
    manager = CanaryManager(rollback_manager=rollback)
    slow = MockModelService("v2.0.0", latency_ms=1200.0)

    dep = manager.start_canary("fraud-graph", "v2.0.0", "v1.9.0", _canary_config())
    dep = manager.step(dep.deployment_id, slow.health)

    assert dep.phase is CanaryPhase.ROLLED_BACK
    assert rollback.list_rollbacks()[0].trigger.severity == "high"


def test_canary_failure_without_auto_rollback_is_reported_not_reverted() -> None:
    """``auto_rollback=False`` must surface FAILED, not silently revert."""
    manager = CanaryManager()
    broken = MockModelService("v2.0.0", error_rate=0.4)

    dep = manager.start_canary(
        "fraud-graph", "v2.0.0", "v1.9.0", _canary_config(auto_rollback=False)
    )
    dep = manager.step(dep.deployment_id, broken.health)

    assert dep.phase is CanaryPhase.FAILED
    assert dep.error is not None and "Health check failed" in dep.error


def test_canary_treats_an_unreachable_mock_service_as_unhealthy() -> None:
    """A raising probe is a failure signal, not an unhandled exception."""
    manager = CanaryManager()
    down = MockModelService("v2.0.0", reachable=False)

    dep = manager.start_canary("fraud-graph", "v2.0.0", "v1.9.0", _canary_config())
    dep = manager.step(dep.deployment_id, down.health)

    assert dep.phase is CanaryPhase.ROLLED_BACK
    assert dep.steps[0].healthy is False


# ---------------------------------------------------------------------------
# Blue-green
# ---------------------------------------------------------------------------


def test_blue_green_switches_only_after_the_green_mock_passes() -> None:
    """Green is tested before it receives live traffic, then takes over."""
    manager = BlueGreenManager()
    green = MockModelService("v2.0.0", latency_ms=45.0)

    dep = manager.prepare("fraud-graph", "v2.0.0", "v1.9.0")
    assert dep.phase is BGPhase.PREPARING

    dep = manager.test_green(dep.deployment_id, green.health)
    assert dep.phase is BGPhase.TESTING
    assert dep.health_check_results[0]["healthy"] is True

    dep = manager.switch(dep.deployment_id)
    assert dep.phase is BGPhase.COMPLETED
    assert dep.blue_version == "v2.0.0"

    dep = manager.monitor(dep.deployment_id, green.health)
    assert dep.phase is BGPhase.MONITORING


def test_blue_green_refuses_to_switch_a_failing_green() -> None:
    """A failing green exhausts its retries and blocks the switch."""
    rollback = RollbackManager()
    manager = BlueGreenManager(rollback_manager=rollback)
    bad_green = MockModelService("v2.0.0", error_rate=0.5)

    config = BlueGreenConfig(auto_switch=False, max_retries=3)
    dep = manager.prepare("fraud-graph", "v2.0.0", "v1.9.0", config)
    dep = manager.test_green(dep.deployment_id, bad_green.health)

    assert dep.phase is BGPhase.FAILED
    assert dep.active_env.value == "blue"
    assert len(dep.health_check_results) == 3
    with pytest.raises(ValueError, match="cannot switch"):
        manager.switch(dep.deployment_id)


def test_blue_green_monitors_and_rolls_back_a_degrading_active_env() -> None:
    """Post-switch monitoring reverts to the previous version on degradation."""
    rollback = RollbackManager()
    manager = BlueGreenManager(rollback_manager=rollback)
    green = MockModelService("v2.0.0")

    dep = manager.prepare("fraud-graph", "v2.0.0", "v1.9.0")
    dep = manager.test_green(dep.deployment_id, green.health)
    dep = manager.switch(dep.deployment_id)

    degraded = MockModelService("v2.0.0", error_rate=0.6, latency_ms=900.0)
    dep = manager.monitor(dep.deployment_id, degraded.health)

    assert dep.phase is BGPhase.ROLLED_BACK
    assert dep.rollback_count == 1
    assert dep.blue_version == "v1.9.0"


def test_blue_green_auto_switch_promotes_a_healthy_green_immediately() -> None:
    """``auto_switch`` skips the manual switch step once green is verified."""
    manager = BlueGreenManager()
    green = MockModelService("v2.0.0")

    dep = manager.prepare("fraud-graph", "v2.0.0", "v1.9.0", BlueGreenConfig(auto_switch=True))
    dep = manager.test_green(dep.deployment_id, green.health)

    assert dep.phase is BGPhase.COMPLETED
    assert dep.blue_version == "v2.0.0"


# ---------------------------------------------------------------------------
# Traffic splitting
# ---------------------------------------------------------------------------


def test_traffic_router_honours_canary_weights_under_a_mock_load() -> None:
    """The split the operator asked for is the split requests actually see."""
    router = TrafficRouter(seed=42)
    rule = router.add_rule(
        "fraud-graph",
        [
            RouteTarget("stable", "v1.9.0", weight=0.9),
            RouteTarget("canary", "v2.0.0", weight=0.1),
        ],
    )

    hits = {"stable": 0, "canary": 0}
    for _ in range(2000):
        target = router.route(rule.rule_id, strategy="weighted")
        assert target is not None
        hits[target.name] += 1

    assert 120 <= hits["canary"] <= 280
    assert sum(hits.values()) == 2000


def test_traffic_router_canary_strategy_falls_back_to_stable_at_zero_weight() -> None:
    """A canary at 0% must receive nothing — no stray traffic."""
    router = TrafficRouter(seed=7)
    rule = router.add_rule(
        "fraud-graph",
        [
            RouteTarget("stable", "v1.9.0", weight=1.0),
            RouteTarget("canary", "v2.0.0", weight=0.0),
        ],
    )

    routed: set[str] = set()
    for _ in range(50):
        target = router.route(rule.rule_id, strategy="canary")
        assert target is not None
        routed.add(target.name)

    assert routed == {"stable"}
