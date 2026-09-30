"""Recommendation engine for cost optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from astroml.infrastructure.cost_optimizer import CostAllocation
from astroml.infrastructure.resource_analyzer import ResourceAnalyzer, WorkloadResourceProfile


@dataclass
class OptimizationRecommendation:
    """A specific cost optimization recommendation."""

    workload_id: str
    recommendation_type: str  # e.g., 'spot_instance', 'auto_scaling', 'right_sizing'
    description: str
    estimated_monthly_savings: float


class RecommendationEngine:
    """Generates cost-saving recommendations for ML workloads."""

    def __init__(self, resource_analyzer: ResourceAnalyzer) -> None:
        self.resource_analyzer = resource_analyzer
        # Spot instance discount rate is typically around 70% cheaper
        self.spot_discount = 0.70

    def add_spot_instance_recommendation(
        self, profile: WorkloadResourceProfile, allocation: CostAllocation
    ) -> OptimizationRecommendation | None:
        """
        Recommend spot instances for batch or non-critical workloads.
        Assume 'batch' workloads are interruptible.
        """
        if profile.workload_type.lower() == "batch":
            monthly_cost = allocation.total_cost * (
                730 / (profile.duration_seconds / 3600.0) if profile.duration_seconds > 0 else 0
            )
            # If the workload only ran once, let's scale it to monthly based on an assumed schedule,
            # or simply use the total cost as a baseline if it's continuous.
            # To simplify, we calculate savings per 730 hours (1 month) based on the hourly rate.
            monthly_savings = allocation.hourly_rate * 730 * self.spot_discount

            return OptimizationRecommendation(
                workload_id=profile.workload_id,
                recommendation_type="spot_instance",
                description=f"Use Spot Instances for batch workload {profile.workload_id} to save up to 70%.",
                estimated_monthly_savings=monthly_savings,
            )
        return None

    def implement_auto_scaling_optimization(
        self, profile: WorkloadResourceProfile, allocation: CostAllocation
    ) -> OptimizationRecommendation | None:
        """
        Recommend auto-scaling if the workload shows spiky or low utilization,
        particularly for inference endpoints.
        """
        scores = self.resource_analyzer.analyze_utilization(profile)
        # If max utilization is below 40% and it's an inference workload, recommend auto-scaling down
        if profile.workload_type.lower() == "inference" and max(scores.values()) < 0.4:
            # Assume auto-scaling can save 30% of the cost by spinning down during idle times
            monthly_savings = allocation.hourly_rate * 730 * 0.30
            return OptimizationRecommendation(
                workload_id=profile.workload_id,
                recommendation_type="auto_scaling",
                description=f"Enable auto-scaling for inference workload {profile.workload_id} due to low average utilization.",
                estimated_monthly_savings=monthly_savings,
            )
        return None

    def add_savings_opportunity_identification(
        self, profile: WorkloadResourceProfile, allocation: CostAllocation
    ) -> OptimizationRecommendation | None:
        """
        Identify general savings opportunities, such as right-sizing an instance.
        """
        scores = self.resource_analyzer.analyze_utilization(profile)
        # If all utilizations are extremely low (<20%), recommend a smaller instance
        if all(score < 0.2 for score in scores.values()):
            # Assume downsizing halves the cost
            monthly_savings = allocation.hourly_rate * 730 * 0.50
            return OptimizationRecommendation(
                workload_id=profile.workload_id,
                recommendation_type="right_sizing",
                description=f"Downsize instance {profile.instance_type} for workload {profile.workload_id} due to <20% utilization.",
                estimated_monthly_savings=monthly_savings,
            )
        return None

    def generate_all_recommendations(
        self, profiles: List[WorkloadResourceProfile], allocations: dict[str, CostAllocation]
    ) -> List[OptimizationRecommendation]:
        """Generate all possible recommendations for the given workloads."""
        recommendations = []
        for profile in profiles:
            allocation = allocations.get(profile.workload_id)
            if not allocation:
                continue

            spot_rec = self.add_spot_instance_recommendation(profile, allocation)
            if spot_rec:
                recommendations.append(spot_rec)

            as_rec = self.implement_auto_scaling_optimization(profile, allocation)
            if as_rec:
                recommendations.append(as_rec)

            rs_rec = self.add_savings_opportunity_identification(profile, allocation)
            if rs_rec:
                recommendations.append(rs_rec)

        return recommendations
