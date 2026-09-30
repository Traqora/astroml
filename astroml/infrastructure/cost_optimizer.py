"""Cost optimization and allocation for ML workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from astroml.infrastructure.resource_analyzer import WorkloadResourceProfile


@dataclass
class CostAllocation:
    """Cost allocation for a specific workload."""

    workload_id: str
    instance_type: str
    duration_hours: float
    hourly_rate: float
    total_cost: float


class CostOptimizer:
    """Calculates costs and optimizations for ML workloads."""

    # Mock pricing table (USD/hr)
    PRICING_TABLE = {
        "ml.m5.large": 0.115,
        "ml.m5.xlarge": 0.23,
        "ml.c5.xlarge": 0.204,
        "ml.p3.2xlarge": 3.825,
        "ml.g4dn.xlarge": 0.736,
    }

    def __init__(self) -> None:
        pass

    def get_hourly_rate(self, instance_type: str) -> float:
        """Get the hourly rate for an instance type. Defaults to a standard rate if unknown."""
        return self.PRICING_TABLE.get(instance_type, 0.50)

    def calculate_workload_cost(self, profile: WorkloadResourceProfile) -> CostAllocation:
        """Calculate the cost allocation for a single workload profile."""
        duration_hours = profile.duration_seconds / 3600.0
        hourly_rate = self.get_hourly_rate(profile.instance_type)
        total_cost = duration_hours * hourly_rate

        return CostAllocation(
            workload_id=profile.workload_id,
            instance_type=profile.instance_type,
            duration_hours=duration_hours,
            hourly_rate=hourly_rate,
            total_cost=total_cost,
        )

    def build_cost_allocation(
        self, profiles: List[WorkloadResourceProfile]
    ) -> Dict[str, CostAllocation]:
        """Build cost allocations for a list of workloads."""
        allocations = {}
        for profile in profiles:
            allocations[profile.workload_id] = self.calculate_workload_cost(profile)
        return allocations

    def calculate_total_cost(self, allocations: Dict[str, CostAllocation]) -> float:
        """Calculate the total cost across all allocations."""
        return sum(allocation.total_cost for allocation in allocations.values())
