#!/usr/bin/env python3
"""Simulate a model deployment strategy against an in-process mock endpoint.

``.github/workflows/model-deployment.yml`` uses this to validate rollout logic on
a runner that has no cluster: the real managers are driven end to end over a
simulated serving endpoint, so traffic shifting, health gating, metrics
capture, automated rollback and the approval gate all execute for real.

Exit codes:
    0  the rollout reached the outcome asked for (promoted, or rolled back with
       ``--expect-rollback``) and the approval gate was satisfied.
    1  the rollout did not reach that outcome.
    2  the strategy or arguments are invalid.
    3  approval gate: no approver was supplied for a rollout that needs one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from astroml.deployment.blue_green import BGPhase, BlueGreenConfig, BlueGreenManager
from astroml.deployment.canary import CanaryConfig, CanaryDeployment, CanaryManager, CanaryPhase
from astroml.deployment.gitops_manager import (
    DeploymentPhase,
    GitOpsManager,
    HealthStatus,
    SyncResult,
    SyncStatus,
)
from astroml.deployment.rollback_manager import RollbackManager
from astroml.deployment.traffic_router import RouteTarget, TrafficRouter

STRATEGIES = ("canary", "blue-green", "direct")

# Never let a misconfigured increment turn the ramp loop into a hang.
MAX_RAMP_STEPS = 40


class SimulatedEndpoint:
    """Stand-in for the model version a rollout would probe.

    Args:
        version: Version string the endpoint serves.
        failure_rate: Fraction of requests that fail.
        latency_ms: Average latency reported per probe.
        fail_after_probes: Degrade hard after this many probes, or never.
    """

    def __init__(
        self,
        version: str,
        failure_rate: float = 0.0,
        latency_ms: float = 25.0,
        fail_after_probes: int | None = None,
    ) -> None:
        self.version = version
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.fail_after_probes = fail_after_probes
        self._probes = 0

    def health(self) -> dict[str, Any]:
        """Return the payload a health probe would read.

        Returns:
            ``healthy`` plus the error-rate and latency metrics a rollout gates
            on, degraded once ``fail_after_probes`` probes have been served.
        """
        self._probes += 1
        degraded = self.fail_after_probes is not None and self._probes > self.fail_after_probes
        error_rate = self.failure_rate if not degraded else max(self.failure_rate, 0.5)
        latency_ms = self.latency_ms if not degraded else max(self.latency_ms, 5000.0)
        return {
            "healthy": error_rate == 0.0,
            "error_rate": error_rate,
            "latency_ms": latency_ms,
            "version": self.version,
        }


def _sync_weights(router: TrafficRouter, rule_id: str, canary_weight: float) -> None:
    """Point the router's weights at the canary's current traffic share.

    Args:
        router: Router serving the model.
        rule_id: Rule governing the model.
        canary_weight: Canary traffic percentage (0-100).
    """
    canary = max(0.0, min(canary_weight, 100.0)) / 100.0
    router.update_weights(rule_id, {"stable": 1.0 - canary, "canary": canary})


def _start_router(model_name: str) -> tuple[TrafficRouter, str]:
    """Create a router with a stable + canary target pair.

    Args:
        model_name: Model the rule serves.

    Returns:
        ``(router, rule_id)``.
    """
    router = TrafficRouter(seed=42)
    rule = router.add_rule(
        model_name,
        [
            RouteTarget("stable", "stable", weight=1.0),
            RouteTarget("canary", "canary", weight=0.0),
        ],
    )
    return router, rule.rule_id


def run_canary(
    endpoint: SimulatedEndpoint,
    model_name: str,
    stable_version: str,
    config: CanaryConfig,
) -> tuple[CanaryManager, CanaryDeployment, TrafficRouter, RollbackManager]:
    """Ramp a canary to promotion, or until it is rolled back.

    Args:
        endpoint: Simulated serving endpoint for the new version.
        model_name: Model being deployed.
        stable_version: Version the canary replaces.
        config: Ramp configuration.

    Returns:
        The manager, the final deployment, the router that carried the shifted
        traffic, and the rollback manager holding any triggered rollback.
    """
    rollback_manager = RollbackManager()
    manager = CanaryManager(rollback_manager=rollback_manager)
    router, rule_id = _start_router(model_name)

    dep = manager.start_canary(model_name, endpoint.version, stable_version, config)
    for _ in range(MAX_RAMP_STEPS):
        if dep.phase not in (CanaryPhase.DEPLOYING, CanaryPhase.RAMPING):
            break
        dep = manager.step(dep.deployment_id, endpoint.health)
        _sync_weights(router, rule_id, dep.current_weight)

    return manager, dep, router, rollback_manager


def run_blue_green(
    endpoint: SimulatedEndpoint, model_name: str, stable_version: str
) -> tuple[BGPhase, int, list[dict[str, Any]]]:
    """Prepare, test, switch and monitor a blue-green release.

    Args:
        endpoint: Simulated serving endpoint for the green version.
        model_name: Model being deployed.
        stable_version: Version currently live on blue.

    Returns:
        ``(final_phase, rollback_count, health_checks)``.
    """
    rollback_manager = RollbackManager()
    manager = BlueGreenManager(rollback_manager=rollback_manager)
    config = BlueGreenConfig(auto_switch=False, max_retries=3, stabilization_seconds=0.0)

    dep = manager.prepare(model_name, endpoint.version, stable_version, config)
    dep = manager.test_green(dep.deployment_id, endpoint.health)
    if dep.phase is BGPhase.TESTING:
        dep = manager.switch(dep.deployment_id)
        dep = manager.monitor(dep.deployment_id, endpoint.health)

    return dep.phase, dep.rollback_count, dep.health_check_results


def approval_gate(
    approver: str,
    initiated_by: str,
    version: str,
    environment: str,
    succeeded: bool,
) -> DeploymentPhase:
    """Record the deployment and require a named approver before it completes.

    Args:
        approver: Actor who approved the release; empty means unapproved.
        initiated_by: Actor who started the deployment.
        version: Model version being released.
        environment: Target environment name.
        succeeded: Whether the rollout reached a healthy terminal state.

    Returns:
        Final :class:`DeploymentPhase` of the recorded deployment.

    Raises:
        PermissionError: When ``approver`` is empty.
    """
    if not approver:
        raise PermissionError("deployment is not approved: --approver is required")

    manager = GitOpsManager(
        argocd_server="https://argocd.invalid.local",
        auth_token="simulation",
        application_name="astroml-model-deployment",
    )
    deployment_id = f"sim-{version}-{int(datetime.now(timezone.utc).timestamp())}"
    manager.create_deployment(
        deployment_id=deployment_id,
        model_version=version,
        image_tag=version,
        environment=environment,
        initiated_by=initiated_by,
    )
    record = manager.approve_deployment(deployment_id, approved_by=approver)

    return manager.complete_deployment(
        deployment_id,
        SyncResult(
            application="astroml-model-deployment",
            sync_status=SyncStatus.SYNCED if succeeded else SyncStatus.UNKNOWN,
            health_status=HealthStatus.HEALTHY if succeeded else HealthStatus.DEGRADED,
            revision=version,
            synced_at=datetime.now(timezone.utc),
            resources_synced=1,
            message="simulated rollout",
        ),
    ).phase


def _print_step_metrics(dep: CanaryDeployment) -> None:
    """Print the canary ramp as a markdown table for the run summary.

    Args:
        dep: Canary deployment whose steps should be listed.
    """
    print("| step | weight % | error rate | latency ms | duration s | healthy |")
    print("|------|----------|------------|------------|------------|---------|")
    for step in dep.steps:
        print(
            f"| {step.step_number} | {step.weight:.1f} | {step.error_rate:.4f} | "
            f"{step.avg_latency_ms:.1f} | {step.duration_seconds:.3f} | {step.healthy} |"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Simulate a model deployment strategy against a mock endpoint."
    )
    parser.add_argument("--strategy", choices=STRATEGIES, default="canary")
    parser.add_argument("--model-name", default="astroml-fraud-graph")
    parser.add_argument("--version", default="v2.0.0")
    parser.add_argument("--stable-version", default="v1.9.0")
    parser.add_argument("--failure-rate", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=25.0)
    parser.add_argument(
        "--fail-after-probes",
        type=int,
        default=None,
        help="Degrade the endpoint after this many probes (failure detection).",
    )
    parser.add_argument(
        "--expect-rollback",
        action="store_true",
        help="Succeed only when the rollout is rolled back instead of promoted.",
    )
    parser.add_argument("--approver", default="", help="Named approver for the gate.")
    parser.add_argument("--initiated-by", default="github-actions")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--initial-weight", type=float, default=5.0)
    parser.add_argument("--increment-step", type=float, default=25.0)
    parser.add_argument("--max-canary-weight", type=float, default=50.0)
    parser.add_argument("--failure-threshold", type=float, default=0.05)
    parser.add_argument("--latency-threshold-ms", type=float, default=500.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the simulated deployment and report whether it behaved.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; see the module docstring.
    """
    args = _build_parser().parse_args(argv)
    endpoint = SimulatedEndpoint(
        version=args.version,
        failure_rate=args.failure_rate,
        latency_ms=args.latency_ms,
        fail_after_probes=args.fail_after_probes,
    )

    print(f"## Deployment simulation: {args.strategy} {args.model_name} {args.version}")

    manager: CanaryManager | None = None
    dep: CanaryDeployment | None = None
    if args.strategy == "blue-green":
        phase, rollback_count, checks = run_blue_green(
            endpoint, args.model_name, args.stable_version
        )
        # A monitored release stays in MONITORING: the switch held, and that is
        # the healthy terminal state for blue-green.
        healthy_terminal = phase in (BGPhase.COMPLETED, BGPhase.MONITORING)
        rolled_back = phase is BGPhase.ROLLED_BACK
        pending_promotion = False
        terminal_note = (
            "green never took traffic, so there was nothing to revert"
            if phase is BGPhase.FAILED
            else "the switch held"
        )
        print(f"- final phase: **{phase.value}**, health checks={len(checks)}")
        print(f"- rollback count: {rollback_count}")
        for index, result in enumerate(checks, start=1):
            print(
                f"  - probe {index}: healthy={result.get('healthy')} "
                f"error_rate={result.get('error_rate', 0.0):.4f} "
                f"latency_ms={result.get('latency_ms', 0.0):.1f}"
            )
    else:
        auto_promote = args.strategy == "direct"
        config = CanaryConfig(
            initial_weight=100.0 if auto_promote else args.initial_weight,
            increment_step=100.0 if auto_promote else args.increment_step,
            max_canary_weight=100.0 if auto_promote else args.max_canary_weight,
            stabilization_seconds=0.0,
            failure_threshold=args.failure_threshold,
            latency_threshold_ms=args.latency_threshold_ms,
            auto_promote=auto_promote,
            auto_rollback=True,
        )
        manager, dep, router, rollback_manager = run_canary(
            endpoint, args.model_name, args.stable_version, config
        )
        _print_step_metrics(dep)
        rule = router.get_rule_for_model(args.model_name)
        weights = (
            {target.name: round(target.weight, 4) for target in rule.targets}
            if rule is not None
            else {}
        )
        rollbacks = rollback_manager.list_rollbacks()
        print(f"- final phase: **{dep.phase.value}**, weight={dep.current_weight:.1f}%")
        print(f"- routed weights: `{weights}`")
        print(f"- rollback records: {len(rollbacks)}")
        for record in rollbacks:
            print(
                f"  - {record.trigger.severity} / {record.status}: "
                f"{record.trigger.reason} -> {record.trigger.target_version}"
            )
        if dep.error:
            print(f"- error: {dep.error}")

        rolled_back = dep.phase is CanaryPhase.ROLLED_BACK
        # STABILIZING means the ramp held at max traffic and is waiting on a
        # human; promoting it is exactly what the approval gate below decides.
        healthy_terminal = dep.phase in (CanaryPhase.STABILIZING, CanaryPhase.PROMOTED)
        pending_promotion = dep.phase is CanaryPhase.STABILIZING
        terminal_note = f"phase is {dep.phase.value}"

    if args.expect_rollback:
        if not rolled_back:
            print(f"::error::expected a rollback, none happened ({terminal_note})")
            return 1
        # An auto-approved rollback needs no further sign-off; the point of the
        # run was that the failure was detected and reverted.
        print("- failure detected and reverted as expected")
        return 0

    if not healthy_terminal:
        print(f"::error::expected promotion, rollout ended unhealthy ({terminal_note})")
        return 1

    # Approval gates completion: a healthy release still waits for a named
    # approver before it is recorded as synced and promoted.
    if not args.approver:
        print("::warning::deployment is not approved: --approver is required")
        return 3

    if pending_promotion and manager is not None and dep is not None:
        dep = manager.promote(dep.deployment_id)
        print(f"- promoted by `{args.approver}`: phase={dep.phase.value}")

    gate_phase = approval_gate(
        approver=args.approver,
        initiated_by=args.initiated_by,
        version=args.version,
        environment=args.environment,
        succeeded=healthy_terminal,
    )

    print(f"- approval gate: **{gate_phase.value}** by `{args.approver}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
