"""Infrastructure API endpoints for AstroML.

Provides cost optimization and resource analysis information.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from astroml.infrastructure.cost_optimizer import CostAllocation, CostOptimizer
from astroml.infrastructure.recommendations import OptimizationRecommendation, RecommendationEngine
from astroml.infrastructure.resource_analyzer import (
    ResourceAnalyzer,
    ResourceMetrics,
    WorkloadResourceProfile,
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class CostAllocationResponse(BaseModel):
    """Cost allocation for a single workload."""

    workload_id: str
    instance_type: str
    duration_hours: float
    hourly_rate: float
    total_cost: float


class OptimizationRecommendationResponse(BaseModel):
    """Cost saving recommendation."""

    workload_id: str
    recommendation_type: str
    description: str
    estimated_monthly_savings: float


class CostOptimizationSummary(BaseModel):
    """Summary of cost allocations and recommendations."""

    total_cost: float
    allocations: List[CostAllocationResponse]
    recommendations: List[OptimizationRecommendationResponse]


# ---------------------------------------------------------------------------
# Router setup and mock data generation
# ---------------------------------------------------------------------------

router = APIRouter()


def _get_mock_profiles() -> List[WorkloadResourceProfile]:
    """Generate mock profiles for the API responses."""
    return [
        WorkloadResourceProfile(
            workload_id="training-job-123",
            workload_type="training",
            metrics=ResourceMetrics(
                cpu_utilization_percent=85.0,
                memory_utilization_percent=90.0,
                gpu_utilization_percent=95.0,
            ),
            instance_type="ml.p3.2xlarge",
            duration_seconds=7200,
        ),
        WorkloadResourceProfile(
            workload_id="inference-svc-456",
            workload_type="inference",
            metrics=ResourceMetrics(
                cpu_utilization_percent=15.0,
                memory_utilization_percent=25.0,
                gpu_utilization_percent=None,
            ),
            instance_type="ml.m5.xlarge",
            duration_seconds=86400,
        ),
        WorkloadResourceProfile(
            workload_id="batch-job-789",
            workload_type="batch",
            metrics=ResourceMetrics(
                cpu_utilization_percent=45.0,
                memory_utilization_percent=50.0,
                gpu_utilization_percent=None,
            ),
            instance_type="ml.c5.xlarge",
            duration_seconds=3600,
        ),
        WorkloadResourceProfile(
            workload_id="idle-job-000",
            workload_type="training",
            metrics=ResourceMetrics(
                cpu_utilization_percent=5.0,
                memory_utilization_percent=10.0,
                gpu_utilization_percent=0.0,
            ),
            instance_type="ml.g4dn.xlarge",
            duration_seconds=18000,
        ),
    ]


@router.get(
    "/api/v1/infrastructure/cost-optimization",
    response_model=CostOptimizationSummary,
    tags=["infrastructure"],
)
async def get_cost_optimization_summary() -> CostOptimizationSummary:
    """Get a summary of cost allocations and optimization recommendations."""
    profiles = _get_mock_profiles()

    resource_analyzer = ResourceAnalyzer()
    cost_optimizer = CostOptimizer()
    recommendation_engine = RecommendationEngine(resource_analyzer)

    allocations_dict = cost_optimizer.build_cost_allocation(profiles)
    recommendations = recommendation_engine.generate_all_recommendations(profiles, allocations_dict)

    total_cost = cost_optimizer.calculate_total_cost(allocations_dict)

    allocation_responses = [
        CostAllocationResponse(
            workload_id=alloc.workload_id,
            instance_type=alloc.instance_type,
            duration_hours=alloc.duration_hours,
            hourly_rate=alloc.hourly_rate,
            total_cost=alloc.total_cost,
        )
        for alloc in allocations_dict.values()
    ]

    recommendation_responses = [
        OptimizationRecommendationResponse(
            workload_id=rec.workload_id,
            recommendation_type=rec.recommendation_type,
            description=rec.description,
            estimated_monthly_savings=rec.estimated_monthly_savings,
        )
        for rec in recommendations
    ]

    return CostOptimizationSummary(
        total_cost=total_cost,
        allocations=allocation_responses,
        recommendations=recommendation_responses,
    )
