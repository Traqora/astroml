from astroml.utils.exceptions import AstroMLError
"""Load balancer for model serving infrastructure.

Issue #639 Step 2: Implements intelligent request routing across
model inference replicas with multiple balancing strategies.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Load balancing routing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class Replica:
    """Information about a model serving replica/pod.

    Attributes:
        replica_id: Unique identifier.
        address: Host:port or URL.
        capacity: Relative capacity weight (default 1.0).
        healthy: Whether the replica is healthy.
        active_connections: Current number of active connections.
        avg_latency_ms: Exponential moving average of latency.
        last_used: Timestamp of last request routed.
    """

    replica_id: str
    address: str
    capacity: float = 1.0
    healthy: bool = True
    active_connections: int = 0
    avg_latency_ms: float = 0.0
    last_used: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_weight(self) -> float:
        """Weight adjusted by capacity and health."""
        if not self.healthy:
            return 0.0
        return self.capacity

    def update_latency(self, latency_ms: float, alpha: float = 0.3) -> None:
        """Update exponential moving average latency."""
        if self.avg_latency_ms == 0.0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica_id": self.replica_id,
            "address": self.address,
            "capacity": self.capacity,
            "healthy": self.healthy,
            "active_connections": self.active_connections,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "effective_weight": self.effective_weight,
        }


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    replica: Replica | None
    strategy: RoutingStrategy
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def successful(self) -> bool:
        return self.replica is not None and self.replica.healthy

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica_id": self.replica.replica_id if self.replica else None,
            "address": self.replica.address if self.replica else None,
            "strategy": self.strategy.value,
            "reason": self.reason,
            "successful": self.successful,
        }


class LoadBalancer:
    """Intelligent load balancer for model inference replicas.

    Routes inference requests across available replicas using configurable
    strategies. Tracks replica health, latency, and connection counts.

    Example:
        lb = LoadBalancer(strategy=RoutingStrategy.LEAST_CONNECTIONS)

        # Register replicas
        lb.register_replica(Replica("r1", "10.0.1.1:8080", capacity=2.0))
        lb.register_replica(Replica("r2", "10.0.1.2:8080", capacity=1.0))

        # Route a request
        decision = lb.route()
        if decision.successful:
            send_request(decision.replica.address, payload)
            lb.release(decision.replica.replica_id)
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
        health_check_fn: Callable[[Replica], bool] | None = None,
    ) -> None:
        """Initialize load balancer.

        Args:
            strategy: Routing strategy to use.
            health_check_fn: Optional function to check replica health.
        """
        self.strategy = strategy
        self._health_check_fn = health_check_fn
        self._replicas: dict[str, Replica] = {}
        self._round_robin_index: int = 0
        self._lock = threading.Lock()
        self._routing_history: deque[RoutingDecision] = deque(maxlen=1000)

    # ── Replica management ──────────────────────────────────────────────

    def register_replica(self, replica: Replica) -> None:
        """Register a new replica."""
        with self._lock:
            self._replicas[replica.replica_id] = replica
            logger.info(f"Registered replica {replica.replica_id} at {replica.address}")

    def deregister_replica(self, replica_id: str) -> bool:
        """Remove a replica."""
        with self._lock:
            if replica_id in self._replicas:
                del self._replicas[replica_id]
                logger.info(f"Deregistered replica {replica_id}")
                return True
            return False

    def mark_healthy(self, replica_id: str) -> None:
        """Mark a replica as healthy."""
        with self._lock:
            if replica_id in self._replicas:
                self._replicas[replica_id].healthy = True

    def mark_unhealthy(self, replica_id: str, reason: str = "") -> None:
        """Mark a replica as unhealthy."""
        with self._lock:
            if replica_id in self._replicas:
                self._replicas[replica_id].healthy = False
                logger.warning(f"Replica {replica_id} marked unhealthy: {reason}")

    def update_metrics(
        self,
        replica_id: str,
        latency_ms: float | None = None,
        active_connections: int | None = None,
    ) -> None:
        """Update performance metrics for a replica."""
        with self._lock:
            r = self._replicas.get(replica_id)
            if r is None:
                return
            if latency_ms is not None:
                r.update_latency(latency_ms)
            if active_connections is not None:
                r.active_connections = active_connections

    @property
    def healthy_replicas(self) -> list[Replica]:
        """Get all healthy replicas."""
        return [r for r in self._replicas.values() if r.healthy]

    @property
    def total_replicas(self) -> int:
        return len(self._replicas)

    @property
    def healthy_count(self) -> int:
        return len(self.healthy_replicas)

    # ── Routing ─────────────────────────────────────────────────────────

    def route(
        self,
        request_context: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> RoutingDecision:
        """Route a request to a replica.

        Args:
            request_context: Optional request-specific info (e.g., for consistent hashing).
            strategy: Override the default routing strategy for this request.

        Returns:
            RoutingDecision with selected replica.
        """
        with self._lock:
            return self._route(request_context, strategy)

    def _route(
        self,
        request_context: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> RoutingDecision:
        healthy = self.healthy_replicas
        if not healthy:
            decision = RoutingDecision(
                replica=None,
                strategy=strategy or self.strategy,
                reason="No healthy replicas available",
            )
            self._routing_history.append(decision)
            return decision

        selected_strategy = strategy or self.strategy

        if selected_strategy == RoutingStrategy.ROUND_ROBIN:
            replica = self._route_round_robin(healthy)
        elif selected_strategy == RoutingStrategy.LEAST_CONNECTIONS:
            replica = self._route_least_connections(healthy)
        elif selected_strategy == RoutingStrategy.LEAST_LATENCY:
            replica = self._route_least_latency(healthy)
        elif selected_strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            replica = self._route_weighted(healthy)
        elif selected_strategy == RoutingStrategy.RANDOM:
            replica = random.choice(healthy) if healthy else None
        elif selected_strategy == RoutingStrategy.CONSISTENT_HASH:
            replica = self._route_consistent_hash(healthy, request_context or {})
        else:
            replica = healthy[0] if healthy else None

        if replica:
            replica.last_used = time.time()
            replica.active_connections += 1

        decision = RoutingDecision(
            replica=replica,
            strategy=selected_strategy,
            reason="",
        )
        self._routing_history.append(decision)
        return decision

    def _route_round_robin(self, replicas: list[Replica]) -> Replica | None:
        """Simple round-robin routing."""
        if not replicas:
            return None
        idx = self._round_robin_index % len(replicas)
        self._round_robin_index += 1
        return replicas[idx]

    def _route_least_connections(self, replicas: list[Replica]) -> Replica | None:
        """Route to replica with fewest active connections."""
        if not replicas:
            return None
        return min(replicas, key=lambda r: r.active_connections)

    def _route_least_latency(self, replicas: list[Replica]) -> Replica | None:
        """Route to replica with lowest average latency."""
        if not replicas:
            return None
        return min(replicas, key=lambda r: r.avg_latency_ms if r.avg_latency_ms > 0 else float("inf"))

    def _route_weighted(self, replicas: list[Replica]) -> Replica | None:
        """Weighted random selection."""
        if not replicas:
            return None
        weights = [r.effective_weight for r in replicas]
        total = sum(weights)
        if total <= 0:
            return random.choice(replicas)

        r = random.random() * total
        cumulative = 0.0
        for replica, weight in zip(replicas, weights):
            cumulative += weight
            if r <= cumulative:
                return replica
        return replicas[-1]

    def _route_consistent_hash(
        self, replicas: list[Replica], context: dict[str, Any]
    ) -> Replica | None:
        """Consistent hashing based on request key."""
        import hashlib

        if not replicas:
            return None

        key = context.get("routing_key", str(time.time()))
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)

        # Simple modulo-based hashing across replicas
        idx = hash_val % len(replicas)
        return replicas[idx]

    def release(self, replica_id: str) -> None:
        """Release a connection from a replica (decrement active count).

        Args:
            replica_id: ID of the replica to release.
        """
        with self._lock:
            r = self._replicas.get(replica_id)
            if r and r.active_connections > 0:
                r.active_connections -= 1

    # ── Health checks ───────────────────────────────────────────────────

    def run_health_checks(self) -> dict[str, bool]:
        """Run health checks on all replicas.

        Returns:
            Dict of replica_id -> healthy status.
        """
        if self._health_check_fn is None:
            return {rid: r.healthy for rid, r in self._replicas.items()}

        results: dict[str, bool] = {}
        for rid, replica in list(self._replicas.items()):
            try:
                healthy = self._health_check_fn(replica)
                with self._lock:
                    replica.healthy = healthy
                results[rid] = healthy
            except AstroMLError as e:
                logger.warning(f"Health check failed for {rid}: {e}")
                with self._lock:
                    replica.healthy = False
                results[rid] = False
        return results

    # ── Stats and diagnostics ───────────────────────────────────────────

    def get_replica_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all replicas."""
        return [r.to_dict() for r in self._replicas.values()]

    def get_overall_stats(self) -> dict[str, Any]:
        """Get aggregate load balancer statistics."""
        return {
            "strategy": self.strategy.value,
            "total_replicas": self.total_replicas,
            "healthy_replicas": self.healthy_count,
            "unhealthy_replicas": self.total_replicas - self.healthy_count,
            "total_active_connections": sum(r.active_connections for r in self._replicas.values()),
            "routing_history_size": len(self._routing_history),
        }

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Change the routing strategy."""
        with self._lock:
            self.strategy = strategy
            logger.info(f"Routing strategy changed to {strategy.value}")


__all__ = [
    "RoutingStrategy",
    "Replica",
    "RoutingDecision",
    "LoadBalancer",
]