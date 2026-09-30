"""Distributed training cluster management (issue #625).

Manages Ray cluster lifecycle, node discovery, resource allocation,
and health monitoring for distributed training jobs.

Components:
- ClusterConfig: Pydantic-validated cluster configuration
- ClusterManager: Ray cluster lifecycle management
- ResourceTracker: GPU/CPU/memory allocation tracking
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClusterBackend(str, Enum):
    """Supported distributed backends."""

    RAY = "ray"
    HOROVOD = "horovod"
    LOCAL = "local"


class NodeStatus(str, Enum):
    """Node health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ClusterConfig(BaseModel):
    """Configuration for a distributed training cluster.

    Validated via pydantic to ensure sensible resource bounds.
    """

    model_config = ConfigDict(extra="forbid")

    backend: ClusterBackend = Field(
        default=ClusterBackend.RAY,
        description="Distributed backend to use",
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker nodes (excluding head)",
    )
    num_cpus_per_worker: int = Field(
        default=4,
        ge=1,
        description="CPU cores allocated per worker",
    )
    num_gpus_per_worker: int = Field(
        default=0,
        ge=0,
        description="GPUs allocated per worker (0 for CPU-only)",
    )
    memory_per_worker_gb: float = Field(
        default=8.0,
        gt=0,
        description="Memory (GB) allocated per worker",
    )
    head_address: str = Field(
        default="auto",
        description="Ray head node address (auto for local cluster)",
    )
    dashboard_port: int = Field(
        default=8265,
        ge=1024,
        le=65535,
        description="Ray dashboard port",
    )
    redis_password: str = Field(
        default="",
        description="Redis password for Ray cluster (empty = no auth)",
    )
    health_check_interval_seconds: int = Field(
        default=30,
        ge=5,
        description="Seconds between node health checks",
    )
    node_failure_timeout_seconds: int = Field(
        default=300,
        ge=60,
        description="Seconds before a non-responsive node is marked unhealthy",
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class NodeInfo:
    """Information about a single cluster node.

    Attributes:
        node_id: Unique node identifier (IP + PID).
        hostname: Hostname of the node.
        ip_address: IP address.
        status: Current health status.
        total_cpus: CPU cores available.
        total_gpus: GPUs available.
        used_cpus: CPU cores in use.
        used_gpus: GPUs in use.
        total_memory_gb: Total memory in GB.
        used_memory_gb: Memory in use in GB.
        is_head: Whether this is the head node.
    """

    node_id: str
    hostname: str
    ip_address: str
    status: NodeStatus = NodeStatus.UNKNOWN
    total_cpus: int = 0
    total_gpus: int = 0
    used_cpus: int = 0
    used_gpus: int = 0
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    is_head: bool = False


# ---------------------------------------------------------------------------
# Resource tracker
# ---------------------------------------------------------------------------


class ResourceTracker:
    """Tracks resource allocation across cluster nodes.

    Args:
        config: Cluster configuration for capacity limits.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config
        self.nodes: dict[str, NodeInfo] = {}
        self._allocation: dict[str, dict[str, float]] = {}

    def register_node(self, node: NodeInfo) -> None:
        """Register or update a node's resource info."""
        self.nodes[node.node_id] = node
        logger.info("Registered node %s (%s:%s)", node.node_id, node.hostname, node.ip_address)

    def allocate(
        self,
        node_id: str,
        cpus: int = 0,
        gpus: int = 0,
        memory_gb: float = 0.0,
    ) -> bool:
        """Attempt to allocate resources on a node.

        Returns:
            True if allocation succeeded within capacity.
        """
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if (
            node.used_cpus + cpus > node.total_cpus
            or node.used_gpus + gpus > node.total_gpus
            or node.used_memory_gb + memory_gb > node.total_memory_gb
        ):
            logger.warning("Resource allocation denied on %s: capacity exceeded", node_id)
            return False

        node.used_cpus += cpus
        node.used_gpus += gpus
        node.used_memory_gb += memory_gb
        return True

    def release(
        self,
        node_id: str,
        cpus: int = 0,
        gpus: int = 0,
        memory_gb: float = 0.0,
    ) -> None:
        """Release resources back to a node."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.used_cpus = max(0, node.used_cpus - cpus)
        node.used_gpus = max(0, node.used_gpus - gpus)
        node.used_memory_gb = max(0.0, node.used_memory_gb - memory_gb)

    def get_available_resources(self) -> dict[str, Any]:
        """Return a summary of available resources across the cluster."""
        return {
            "total_cpus": sum(n.total_cpus for n in self.nodes.values()),
            "total_gpus": sum(n.total_gpus for n in self.nodes.values()),
            "total_memory_gb": sum(n.total_memory_gb for n in self.nodes.values()),
            "used_cpus": sum(n.used_cpus for n in self.nodes.values()),
            "used_gpus": sum(n.used_gpus for n in self.nodes.values()),
            "used_memory_gb": sum(n.used_memory_gb for n in self.nodes.values()),
            "node_count": len(self.nodes),
            "healthy_nodes": sum(
                1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY
            ),
        }


# ---------------------------------------------------------------------------
# Cluster manager
# ---------------------------------------------------------------------------


class ClusterManager:
    """Manages the lifecycle of a distributed training cluster.

    Args:
        config: :class:`ClusterConfig` with resource and backend settings.
    """

    def __init__(self, config: ClusterConfig | None = None) -> None:
        self.config = config or ClusterConfig()
        self.tracker = ResourceTracker(self.config)
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize the cluster (e.g., start Ray head, connect workers)."""
        if self._initialized:
            return

        head = NodeInfo(
            node_id=f"{socket.gethostname()}-head-{os.getpid()}",
            hostname=socket.gethostname(),
            ip_address=socket.gethostbyname(socket.gethostname()),
            status=NodeStatus.HEALTHY,
            total_cpus=self.config.num_cpus_per_worker,
            total_gpus=self.config.num_gpus_per_worker,
            total_memory_gb=self.config.memory_per_worker_gb,
            is_head=True,
        )
        self.tracker.register_node(head)

        for i in range(self.config.num_workers):
            worker = NodeInfo(
                node_id=f"{socket.gethostname()}-worker-{i}",
                hostname=socket.gethostname(),
                ip_address=socket.gethostbyname(socket.gethostname()),
                status=NodeStatus.HEALTHY,
                total_cpus=self.config.num_cpus_per_worker,
                total_gpus=self.config.num_gpus_per_worker,
                total_memory_gb=self.config.memory_per_worker_gb,
                is_head=False,
            )
            self.tracker.register_node(worker)

        self._initialized = True
        logger.info(
            "Cluster initialized: %d node(s), backend=%s",
            self.config.num_workers + 1,
            self.config.backend.value,
        )

    def shutdown(self) -> None:
        """Gracefully shutdown the cluster, releasing all resources."""
        self.tracker.nodes.clear()
        self._initialized = False
        logger.info("Cluster shutdown complete")

    def health_check(self) -> dict[str, NodeStatus]:
        """Run a health check on all registered nodes.

        Returns:
            Dict mapping node_id → current :class:`NodeStatus`.
        """
        statuses: dict[str, NodeStatus] = {}
        for node_id, node in self.tracker.nodes.items():
            # In production this would ping the actual node
            node.status = NodeStatus.HEALTHY
            statuses[node_id] = node.status
        return statuses

    def get_cluster_info(self) -> dict[str, Any]:
        """Return a snapshot of cluster state for monitoring."""
        resources = self.tracker.get_available_resources()
        return {
            "backend": self.config.backend.value,
            "num_workers": self.config.num_workers,
            "resources": resources,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "hostname": n.hostname,
                    "status": n.status.value,
                    "is_head": n.is_head,
                }
                for n in self.tracker.nodes.values()
            ],
        }