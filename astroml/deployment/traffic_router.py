"""Traffic router for splitting inference requests between model versions.

Supports weighted random routing, session affinity, and canary-based
traffic shifting for A/B testing and gradual rollouts.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RouteTarget:
    """A model version target for traffic routing.

    Attributes:
        name: Human-readable target name.
        version: Model version string.
        weight: Traffic weight (0.0–1.0).
        endpoint: Optional serving endpoint URL.
        metadata: Arbitrary metadata.
    """

    name: str
    version: str
    weight: float = 1.0
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingRule:
    """A rule controlling traffic routing between targets.

    Attributes:
        rule_id: Unique identifier.
        model_name: Model this rule applies to.
        targets: List of RouteTarget with their weights.
        sticky_sessions: Whether to use session-based affinity.
        default_target: Fallback target name if routing fails.
    """

    rule_id: str
    model_name: str
    targets: list[RouteTarget] = field(default_factory=list)
    sticky_sessions: bool = False
    default_target: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TrafficRouter:
    """Routes inference requests across model versions with configurable weights.

    Supports multiple routing strategies:
    - Weighted random selection
    - Round-robin
    - Sticky sessions (hash-based affinity)
    - Canary traffic shifting

    Thread-safe for concurrent access.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rules: dict[str, RoutingRule] = {}
        self._lock = threading.Lock()
        self._rng = random.Random(seed)
        self._round_robin_counters: dict[str, int] = {}
        # Session affinity store: session_id -> target_name
        self._sessions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        model_name: str,
        targets: list[RouteTarget],
        sticky_sessions: bool = False,
        default_target: str | None = None,
    ) -> RoutingRule:
        """Create a new routing rule for a model.

        Args:
            model_name: Name of the model to route.
            targets: List of route targets with weights.
            sticky_sessions: Whether to enable session affinity.
            default_target: Fallback target name.

        Returns:
            The created RoutingRule.
        """
        import uuid as _uuid

        rule = RoutingRule(
            rule_id=_uuid.uuid4().hex[:8],
            model_name=model_name,
            targets=targets,
            sticky_sessions=sticky_sessions,
            default_target=default_target,
        )
        with self._lock:
            self._rules[rule.rule_id] = rule
            logger.info(
                "Added routing rule %s for model '%s' (%d targets)",
                rule.rule_id,
                model_name,
                len(targets),
            )
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a routing rule.

        Args:
            rule_id: Rule to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
        return False

    def get_rule(self, rule_id: str) -> RoutingRule | None:
        """Get a routing rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            RoutingRule or None.
        """
        with self._lock:
            return self._rules.get(rule_id)

    def get_rule_for_model(self, model_name: str) -> RoutingRule | None:
        """Get the first routing rule for a model.

        Args:
            model_name: Model name.

        Returns:
            RoutingRule or None.
        """
        with self._lock:
            for rule in self._rules.values():
                if rule.model_name == model_name:
                    return rule
        return None

    def update_weights(
        self,
        rule_id: str,
        weights: dict[str, float],
    ) -> RoutingRule:
        """Update traffic weights for targets in a rule.

        Args:
            rule_id: Rule to update.
            weights: Mapping of target name to new weight.

        Returns:
            Updated rule.
        """
        rule = self._get_rule(rule_id)
        with self._lock:
            for target in rule.targets:
                if target.name in weights:
                    target.weight = weights[target.name]
            rule.updated_at = datetime.now(timezone.utc).isoformat()
        return rule

    def list_rules(self) -> list[RoutingRule]:
        """List all routing rules.

        Returns:
            Copy of all rules.
        """
        with self._lock:
            return list(self._rules.values())

    # ------------------------------------------------------------------
    # Routing strategies
    # ------------------------------------------------------------------

    def route(
        self,
        rule_id: str,
        session_id: str | None = None,
        strategy: str = "weighted",
    ) -> RouteTarget | None:
        """Route a request to a target based on the configured strategy.

        Args:
            rule_id: Routing rule to use.
            session_id: Optional session ID for sticky sessions.
            strategy: Routing strategy (``weighted``, ``round_robin``,
                      ``canary``, ``sticky``).

        Returns:
            Selected RouteTarget, or None if routing fails.
        """
        rule = self._get_rule(rule_id)

        if not rule.targets:
            logger.warning("Routing rule %s has no targets", rule_id)
            return None

        # Sticky sessions
        if rule.sticky_sessions and session_id:
            with self._lock:
                if session_id in self._sessions:
                    target_name = self._sessions[session_id]
                    for t in rule.targets:
                        if t.name == target_name:
                            return t

        # Strategy dispatch
        if strategy == "round_robin":
            target = self._route_round_robin(rule)
        elif strategy == "sticky" and session_id:
            target = self._route_sticky(rule, session_id)
        elif strategy == "canary":
            target = self._route_canary(rule)
        else:
            target = self._route_weighted(rule)

        if target is None:
            # Fallback
            if rule.default_target:
                for t in rule.targets:
                    if t.name == rule.default_target:
                        target = t
                        break

        if target is None:
            logger.error(
                "Failed to route request for rule %s, no targets available",
                rule_id,
            )

        return target

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _route_weighted(self, rule: RoutingRule) -> RouteTarget | None:
        """Weighted random selection."""
        targets = [t for t in rule.targets if t.weight > 0]
        if not targets:
            return None

        total = sum(t.weight for t in targets)
        if total <= 0:
            return targets[0]

        r = self._rng.uniform(0, total)
        cumulative = 0.0
        for t in targets:
            cumulative += t.weight
            if r <= cumulative:
                return t

        return targets[-1]

    def _route_round_robin(self, rule: RoutingRule) -> RouteTarget | None:
        """Deterministic round-robin selection."""
        targets = [t for t in rule.targets if t.weight > 0]
        if not targets:
            return None

        with self._lock:
            idx = self._round_robin_counters.get(rule.rule_id, 0)
            self._round_robin_counters[rule.rule_id] = (idx + 1) % len(targets)
            return targets[idx % len(targets)]

    def _route_sticky(
        self, rule: RoutingRule, session_id: str
    ) -> RouteTarget | None:
        """Hash-based session affinity."""
        targets = [t for t in rule.targets if t.weight > 0]
        if not targets:
            return None

        # Deterministic hash -> target
        h = hashlib.sha256(f"{rule.rule_id}:{session_id}".encode()).hexdigest()
        idx = int(h, 16) % len(targets)
        target = targets[idx]

        # Store assignment
        with self._lock:
            self._sessions[session_id] = target.name

        return target

    def _route_canary(self, rule: RoutingRule) -> RouteTarget | None:
        """Route based on canary weight - the lowest-weight target gets
        treated as the canary, the highest-weight target as stable.

        In canary mode, weights represent traffic percentages:
        stable_target receives the complement.
        """
        targets = [t for t in rule.targets if t.weight > 0]
        if not targets:
            return None

        # Sort by weight: stable first (highest weight), canary last
        sorted_targets = sorted(targets, key=lambda t: t.weight, reverse=True)
        if len(sorted_targets) < 2:
            return sorted_targets[0]

        stable = sorted_targets[0]
        canary = sorted_targets[1]

        # Canary weight interpreted as percentage threshold (0–100)
        canary_pct = min(canary.weight * 100, 100.0)
        r = self._rng.uniform(0, 100)

        return canary if r <= canary_pct else stable

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_sessions(self) -> None:
        """Clear all sticky session assignments."""
        with self._lock:
            self._sessions.clear()
            logger.info("Cleared all sticky sessions")

    def remove_session(self, session_id: str) -> None:
        """Remove a specific sticky session.

        Args:
            session_id: Session to remove.
        """
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        """Return the number of active sticky sessions.

        Returns:
            Active session count.
        """
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_rule(self, rule_id: str) -> RoutingRule:
        """Get a rule or raise ValueError."""
        with self._lock:
            rule = self._rules.get(rule_id)
        if rule is None:
            raise ValueError(f"Routing rule '{rule_id}' not found")
        return rule