"""Tests for canary and blue-green deployment strategies."""

import time

from astroml.deployment.blue_green import BlueGreenConfig, BlueGreenManager, BGPhase
from astroml.deployment.canary import (
    CanaryConfig,
    CanaryDeployment,
    CanaryManager,
    CanaryPhase,
)
from astroml.deployment.rollback_manager import RollbackManager
from astroml.deployment.traffic_router import RouteTarget, TrafficRouter


# ---------------------------------------------------------------------------
# CanaryManager
# ---------------------------------------------------------------------------


def test_start_canary() -> None:
    mgr = CanaryManager()
    dep = mgr.start_canary("fraud-model", "v2.0", "v1.0")
    assert dep.model_name == "fraud-model"
    assert dep.canary_version == "v2.0"
    assert dep.stable_version == "v1.0"
    assert dep.phase == CanaryPhase.DEPLOYING
    assert dep.current_weight == 5.0  # default initial


def test_canary_step_healthy() -> None:
    mgr = CanaryManager()
    dep = mgr.start_canary("m", "v2", "v1",
                           CanaryConfig(initial_weight=10, increment_step=10,
                                        max_canary_weight=50))

    def healthy() -> dict:
        return {"error_rate": 0.0, "latency_ms": 50.0}

    dep = mgr.step(dep.deployment_id, healthy)
    assert dep.phase == CanaryPhase.RAMPING
    assert dep.current_weight >= 10
    assert len(dep.steps) == 1
    assert dep.steps[0].healthy is True


def test_canary_step_unhealthy_triggers_rollback() -> None:
    mgr = CanaryManager()
    config = CanaryConfig(
        initial_weight=10,
        failure_threshold=0.01,
        auto_rollback=True,
    )
    dep = mgr.start_canary("m", "v2", "v1", config)

    def unhealthy() -> dict:
        return {"error_rate": 0.10, "latency_ms": 100.0}

    dep = mgr.step(dep.deployment_id, unhealthy)
    assert dep.phase == CanaryPhase.ROLLED_BACK
    assert dep.error is not None


def test_promote_canary() -> None:
    mgr = CanaryManager()
    dep = mgr.start_canary("m", "v2", "v1")
    dep = mgr.promote(dep.deployment_id)
    assert dep.phase == CanaryPhase.PROMOTED
    assert dep.current_weight == 100.0


def test_rollback_canary() -> None:
    mgr = CanaryManager()
    dep = mgr.start_canary("m", "v2", "v1")
    dep = mgr.rollback(dep.deployment_id)
    assert dep.phase == CanaryPhase.ROLLED_BACK
    assert dep.current_weight == 0.0


def test_list_active_canaries() -> None:
    mgr = CanaryManager()
    dep = mgr.start_canary("m1", "v2", "v1")
    mgr.start_canary("m2", "v3", "v2")
    active = mgr.list_active()
    assert len(active) == 2

    mgr.promote(dep.deployment_id)
    active = mgr.list_active()
    assert len(active) == 1


def test_set_weight() -> None:
    mgr = CanaryManager()
    dep = mgr.start_canary("m", "v2", "v1")
    dep = mgr.set_weight(dep.deployment_id, 30.0)
    assert dep.current_weight == 30.0


def test_canary_status_summary() -> None:
    mgr = CanaryManager()
    mgr.start_canary("m", "v2", "v1")
    summary = mgr.status_summary()
    assert summary["total"] == 1
    assert summary["by_phase"]["deploying"] == 1


def test_canary_get_nonexistent() -> None:
    mgr = CanaryManager()
    assert mgr.get("missing") is None


# ---------------------------------------------------------------------------
# BlueGreenManager
# ---------------------------------------------------------------------------


def test_prepare_blue_green() -> None:
    mgr = BlueGreenManager()
    dep = mgr.prepare("fraud-model", "v2.0", "v1.0")
    assert dep.model_name == "fraud-model"
    assert dep.green_version == "v2.0"
    assert dep.blue_version == "v1.0"
    assert dep.phase == BGPhase.PREPARING


def test_test_green_healthy() -> None:
    mgr = BlueGreenManager()
    dep = mgr.prepare("m", "v2", "v1")

    def healthy() -> dict:
        return {"healthy": True}

    dep = mgr.test_green(dep.deployment_id, healthy)
    assert dep.phase in (BGPhase.TESTING, BGPhase.COMPLETED)
    assert len(dep.health_check_results) >= 1
    assert dep.health_check_results[0]["healthy"] is True


def test_test_green_unhealthy() -> None:
    mgr = BlueGreenManager()
    config = BlueGreenConfig(max_retries=1)
    dep = mgr.prepare("m", "v2", "v1", config)

    def unhealthy() -> dict:
        return {"healthy": False, "error": "crash"}

    dep = mgr.test_green(dep.deployment_id, unhealthy)
    assert dep.phase == BGPhase.FAILED
    assert dep.error is not None


def test_switch_blue_green() -> None:
    mgr = BlueGreenManager()
    dep = mgr.prepare("m", "v2", "v1")

    # Manually set to testing so we can switch
    dep.phase = BGPhase.TESTING
    dep = mgr.switch(dep.deployment_id)
    assert dep.phase == BGPhase.COMPLETED
    # After switch: blue became green=v2 (active), green became blue=v1
    assert dep.blue_version == "v2"


def test_rollback_blue_green() -> None:
    mgr = BlueGreenManager()
    dep = mgr.prepare("m", "v2", "v1")
    dep = mgr.rollback(dep.deployment_id)
    assert dep.phase == BGPhase.ROLLED_BACK
    assert dep.rollback_count == 1


def test_monitor_triggers_rollback() -> None:
    mgr = BlueGreenManager()
    config = BlueGreenConfig(auto_rollback=True)
    dep = mgr.prepare("m", "v2", "v1", config)
    dep.phase = BGPhase.COMPLETED

    def unhealthy() -> dict:
        return {"healthy": False}

    dep = mgr.monitor(dep.deployment_id, unhealthy)
    assert dep.phase == BGPhase.ROLLED_BACK


def test_list_deployments() -> None:
    mgr = BlueGreenManager()
    mgr.prepare("m1", "v2", "v1")
    mgr.prepare("m2", "v3", "v2")
    assert len(mgr.list_deployments()) == 2
    assert len(mgr.list_deployments(model_name="m1")) == 1


def test_status_summary() -> None:
    mgr = BlueGreenManager()
    mgr.prepare("m", "v2", "v1")
    summary = mgr.status_summary()
    assert summary["total"] == 1


# ---------------------------------------------------------------------------
# TrafficRouter
# ---------------------------------------------------------------------------


def test_route_weighted() -> None:
    tr = TrafficRouter(seed=42)
    targets = [
        RouteTarget(name="stable", version="v1", weight=0.8),
        RouteTarget(name="canary", version="v2", weight=0.2),
    ]
    rule = tr.add_rule("model", targets)

    # Run many routes and check distribution
    results = {}
    for _ in range(1000):
        t = tr.route(rule.rule_id, strategy="weighted")
        assert t is not None
        results[t.name] = results.get(t.name, 0) + 1

    assert results["stable"] > 700  # ~800
    assert results["canary"] > 100  # ~200


def test_route_round_robin() -> None:
    tr = TrafficRouter()
    targets = [
        RouteTarget(name="a", version="v1", weight=1.0),
        RouteTarget(name="b", version="v2", weight=1.0),
    ]
    rule = tr.add_rule("model", targets)
    t1 = tr.route(rule.rule_id, strategy="round_robin")
    t2 = tr.route(rule.rule_id, strategy="round_robin")
    assert t1 is not None and t2 is not None
    assert t1.name != t2.name


def test_route_sticky() -> None:
    tr = TrafficRouter()
    targets = [
        RouteTarget(name="a", version="v1", weight=1.0),
        RouteTarget(name="b", version="v2", weight=1.0),
    ]
    rule = tr.add_rule("model", targets, sticky_sessions=True)

    session = "user-123"
    t1 = tr.route(rule.rule_id, session_id=session, strategy="sticky")
    # Same session should get same target
    t2 = tr.route(rule.rule_id, session_id=session, strategy="sticky")
    assert t1 is not None and t2 is not None
    assert t1.name == t2.name


def test_route_canary() -> None:
    tr = TrafficRouter(seed=1)
    targets = [
        RouteTarget(name="stable", version="v1", weight=0.8),
        RouteTarget(name="canary", version="v2", weight=0.1),  # 10% canary weight
    ]
    rule = tr.add_rule("model", targets)

    canary_hits = 0
    for _ in range(1000):
        t = tr.route(rule.rule_id, strategy="canary")
        assert t is not None
        if t.name == "canary":
            canary_hits += 1
    # Canary should get roughly ~10%
    assert canary_hits < 200  # Not too many


def test_update_weights() -> None:
    tr = TrafficRouter()
    targets = [
        RouteTarget(name="a", version="v1", weight=0.5),
        RouteTarget(name="b", version="v2", weight=0.5),
    ]
    rule = tr.add_rule("model", targets)
    tr.update_weights(rule.rule_id, {"a": 0.9, "b": 0.1})

    rule = tr.get_rule(rule.rule_id)
    assert rule is not None
    assert rule.targets[0].weight == 0.9


def test_no_targets_returns_none() -> None:
    tr = TrafficRouter()
    rule = tr.add_rule("model", [])
    t = tr.route(rule.rule_id)
    assert t is None


def test_default_target_fallback() -> None:
    tr = TrafficRouter()
    targets = [RouteTarget(name="fallback", version="v1", weight=0.0)]
    rule = tr.add_rule("model", targets, default_target="fallback")

    t = tr.route(rule.rule_id)
    assert t is not None
    assert t.name == "fallback"


def test_session_management() -> None:
    tr = TrafficRouter()
    targets = [RouteTarget(name="a", version="v1", weight=1.0)]
    rule = tr.add_rule("model", targets, sticky_sessions=True)

    tr.route(rule.rule_id, session_id="s1", strategy="sticky")
    tr.route(rule.rule_id, session_id="s2", strategy="sticky")
    assert tr.session_count() == 2
    tr.clear_sessions()
    assert tr.session_count() == 0