from astroml.utils.exceptions import AstroMLError
"""Resource manager for model serving infrastructure.

Issue #639 Step 3: Manages GPU/CPU allocation for model inference pods,
resource reservation, and scheduling.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource specification
# ---------------------------------------------------------------------------


class DeviceType(Enum):
    """Types of compute devices."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass
class ResourceRequest:
    """Resource requirements for a model or inference request.

    Attributes:
        cpu_cores: Number of CPU cores.
        memory_mb: Memory in MB.
        gpu_count: Number of GPUs.
        gpu_memory_mb: GPU memory in MB per GPU.
        gpu_type: Type of GPU (e.g., 'T4', 'A100').
        priority: Priority (lower = higher priority).
    """

    cpu_cores: float = 1.0
    memory_mb: float = 512.0
    gpu_count: int = 0
    gpu_memory_mb: float = 0.0
    gpu_type: str = ""
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_type": self.gpu_type,
            "priority": self.priority,
        }


@dataclass
class ResourceCapacity:
    """Total resource capacity of the serving cluster."""

    total_cpu_cores: float = 0.0
    total_memory_mb: float = 0.0
    total_gpu_count: int = 0
    total_gpu_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cpu_cores": self.total_cpu_cores,
            "total_memory_mb": self.total_memory_mb,
            "total_gpu_count": self.total_gpu_count,
            "total_gpu_memory_mb": self.total_gpu_memory_mb,
        }


@dataclass
class ResourceAllocation:
    """An active resource allocation for a model replica."""

    allocation_id: str
    model_id: str
    replica_id: str
    requested: ResourceRequest
    node_name: str = ""
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "model_id": self.model_id,
            "replica_id": self.replica_id,
            "cpu_cores": self.requested.cpu_cores,
            "memory_mb": self.requested.memory_mb,
            "gpu_count": self.requested.gpu_count,
            "node_name": self.node_name,
            "allocated_at": self.allocated_at.isoformat(),
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Resource Manager
# ---------------------------------------------------------------------------


class ResourceManager:
    """Manages resource allocation and scheduling for model inference.

    Tracks available compute resources (CPU, GPU, memory), manages
    allocation/deallocation for model replicas, and enforces limits.

    Example:
        rm = ResourceManager(
            capacity=ResourceCapacity(
                total_cpu_cores=32,
                total_memory_mb=65536,
                total_gpu_count=4,
                total_gpu_memory_mb=65536,
            ),
        )

        # Request resources for a replica
        alloc = rm.allocate(
            model_id="fraud-detector",
            replica_id="fraud-detector-1",
            request=ResourceRequest(cpu_cores=2, gpu_count=1, gpu_memory_mb=8000),
        )

        if alloc is None:
            print("Insufficient resources")

        # When done:
        rm.deallocate(alloc.allocation_id)
    """

    def __init__(self, capacity: ResourceCapacity | None = None) -> None:
        """Initialize resource manager.

        Args:
            capacity: Total cluster capacity. Uses defaults if None.
        """
        self.capacity = capacity or ResourceCapacity()
        self._allocations: dict[str, ResourceAllocation] = {}
        self._lock = threading.Lock()
        self._resource_usage_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._events: list[dict[str, Any]] = []

    @property
    def allocated_cpu(self) -> float:
        """Total CPU cores currently allocated."""
        return sum(
            a.requested.cpu_cores for a in self._allocations.values() if a.active
        )

    @property
    def allocated_memory_mb(self) -> float:
        """Total memory currently allocated."""
        return sum(
            a.requested.memory_mb for a in self._allocations.values() if a.active
        )

    @property
    def allocated_gpu_count(self) -> int:
        """Total GPU count currently allocated."""
        return sum(
            a.requested.gpu_count for a in self._allocations.values() if a.active
        )

    @property
    def available_cpu(self) -> float:
        """Available CPU cores."""
        return max(0.0, self.capacity.total_cpu_cores - self.allocated_cpu)

    @property
    def available_memory_mb(self) -> float:
        """Available memory in MB."""
        return max(0.0, self.capacity.total_memory_mb - self.allocated_memory_mb)

    @property
    def available_gpu_count(self) -> int:
        """Available GPU count."""
        return max(0, self.capacity.total_gpu_count - self.allocated_gpu_count)

    def can_allocate(self, request: ResourceRequest) -> bool:
        """Check if there are sufficient resources for a request.

        Args:
            request: Resource requirements to check.

        Returns:
            True if resources can be allocated.
        """
        with self._lock:
            return (
                request.cpu_cores <= self.available_cpu
                and request.memory_mb <= self.available_memory_mb
                and request.gpu_count <= self.available_gpu_count
                and request.gpu_memory_mb <= self.capacity.total_gpu_memory_mb - self.allocated_gpu_memory()
            )

    @property
    def allocated_gpu_memory(self) -> float:
        """Total GPU memory allocated in MB."""
        return sum(
            a.requested.gpu_memory_mb * a.requested.gpu_count
            for a in self._allocations.values() if a.active
        )

    def allocate(
        self,
        model_id: str,
        replica_id: str,
        request: ResourceRequest,
        node_name: str = "",
    ) -> ResourceAllocation | None:
        """Allocate resources for a model replica.

        Args:
            model_id: Model identifier.
            replica_id: Replica identifier.
            request: Resource requirements.
            node_name: Target node name (optional).

        Returns:
            ResourceAllocation if successful, None if insufficient resources.
        """
        import uuid

        with self._lock:
            if not self.can_allocate(request):
                logger.warning(
                    f"Cannot allocate for {model_id}/{replica_id}: "
                    f"need {request.cpu_cores} CPU, {request.memory_mb}MB RAM, "
                    f"{request.gpu_count} GPU (available: {self.available_cpu:.1f} CPU, "
                    f"{self.available_memory_mb:.0f}MB RAM, {self.available_gpu_count} GPU)"
                )
                return None

            allocation = ResourceAllocation(
                allocation_id=uuid.uuid4().hex[:12],
                model_id=model_id,
                replica_id=replica_id,
                requested=request,
                node_name=node_name,
            )
            self._allocations[allocation.allocation_id] = allocation

            self._events.append({
                "type": "allocate",
                "model_id": model_id,
                "replica_id": replica_id,
                "allocation_id": allocation.allocation_id,
                "cpu": request.cpu_cores,
                "gpu": request.gpu_count,
                "timestamp": time.time(),
            })

            logger.info(
                f"Allocated {request.cpu_cores} CPU, {request.memory_mb}MB RAM, "
                f"{request.gpu_count} GPU for {model_id}/{replica_id}"
                f" (available: {self.available_cpu:.1f} CPU, {self.available_gpu_count} GPU)"
            )

            self._notify_callbacks()
            return allocation

    def deallocate(self, allocation_id: str) -> bool:
        """Release resources for an allocation.

        Args:
            allocation_id: ID of the allocation to release.

        Returns:
            True if deallocated successfully.
        """
        with self._lock:
            alloc = self._allocations.get(allocation_id)
            if alloc is None:
                return False
            if not alloc.active:
                return False

            alloc.active = False
            self._events.append({
                "type": "deallocate",
                "model_id": alloc.model_id,
                "replica_id": alloc.replica_id,
                "allocation_id": allocation_id,
                "timestamp": time.time(),
            })

            logger.info(
                f"Deallocated resources for {alloc.model_id}/{alloc.replica_id}: "
                f"{alloc.requested.cpu_cores} CPU, {alloc.requested.gpu_count} GPU freed"
            )
            self._notify_callbacks()
            return True

    def deallocate_by_model(self, model_id: str) -> int:
        """Release all allocations for a model.

        Args:
            model_id: Model to deallocate.

        Returns:
            Number of allocations released.
        """
        count = 0
        with self._lock:
            for alloc in list(self._allocations.values()):
                if alloc.model_id == model_id and alloc.active:
                    alloc.active = False
                    count += 1
        if count > 0:
            self._notify_callbacks()
        return count

    def deallocate_by_replica(self, replica_id: str) -> bool:
        """Release allocation for a specific replica."""
        with self._lock:
            for alloc in self._allocations.values():
                if alloc.replica_id == replica_id and alloc.active:
                    alloc.active = False
                    self._notify_callbacks()
                    return True
        return False

    def get_model_allocations(self, model_id: str) -> list[ResourceAllocation]:
        """Get all active allocations for a model."""
        return [a for a in self._allocations.values() if a.model_id == model_id and a.active]

    def get_replica_allocation(self, replica_id: str) -> ResourceAllocation | None:
        """Get allocation for a specific replica."""
        for alloc in self._allocations.values():
            if alloc.replica_id == replica_id and alloc.active:
                return alloc
        return None

    def get_utilization(self) -> dict[str, dict[str, float]]:
        """Get current resource utilization.

        Returns:
            Dict with 'cpu', 'memory', 'gpu' keys, each with
            'total', 'used', 'available', 'utilization_pct'.
        """
        with self._lock:
            cpu_used = self.allocated_cpu
            mem_used = self.allocated_memory_mb
            gpu_used = float(self.allocated_gpu_count)

            cpu_total = self.capacity.total_cpu_cores
            mem_total = self.capacity.total_memory_mb
            gpu_total = float(self.capacity.total_gpu_count)

            return {
                "cpu": {
                    "total": cpu_total,
                    "used": cpu_used,
                    "available": max(0.0, cpu_total - cpu_used),
                    "utilization_pct": round(cpu_used / cpu_total * 100, 1) if cpu_total > 0 else 0.0,
                },
                "memory": {
                    "total": mem_total,
                    "used": mem_used,
                    "available": max(0.0, mem_total - mem_used),
                    "utilization_pct": round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0.0,
                },
                "gpu": {
                    "total": gpu_total,
                    "used": gpu_used,
                    "available": max(0.0, gpu_total - gpu_used),
                    "utilization_pct": round(gpu_used / gpu_total * 100, 1) if gpu_total > 0 else 0.0,
                },
            }

    def get_dashboard(self) -> dict[str, Any]:
        """Get a dashboard summary."""
        util = self.get_utilization()
        return {
            "capacity": self.capacity.to_dict(),
            "utilization": util,
            "active_allocations": sum(1 for a in self._allocations.values() if a.active),
            "total_allocations": len(self._allocations),
        }

    def add_usage_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked when resource usage changes."""
        self._resource_usage_callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of usage changes."""
        usage = self.get_utilization()
        for cb in self._resource_usage_callbacks:
            try:
                cb(usage)
            except AstroMLError as e:
                logger.warning(f"Resource usage callback failed: {e}")

    def update_capacity(self, capacity: ResourceCapacity) -> None:
        """Update total cluster capacity."""
        with self._lock:
            self.capacity = capacity
            logger.info(f"Updated capacity: {capacity.to_dict()}")


__all__ = [
    "DeviceType",
    "ResourceRequest",
    "ResourceCapacity",
    "ResourceAllocation",
    "ResourceManager",
]