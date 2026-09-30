import pytest

from astroml.infrastructure.cost_optimizer import CostAllocation, CostOptimizer
from astroml.infrastructure.recommendations import RecommendationEngine
from astroml.infrastructure.resource_analyzer import (
    ResourceAnalyzer,
    ResourceMetrics,
    WorkloadResourceProfile,
)


@pytest.fixture
def resource_analyzer():
    return ResourceAnalyzer()


@pytest.fixture
def cost_optimizer():
    return CostOptimizer()


@pytest.fixture
def recommendation_engine(resource_analyzer):
    return RecommendationEngine(resource_analyzer)


@pytest.fixture
def mock_profiles():
    return [
        WorkloadResourceProfile(
            workload_id="test-batch-1",
            workload_type="batch",
            metrics=ResourceMetrics(cpu_utilization_percent=50.0, memory_utilization_percent=60.0),
            instance_type="ml.c5.xlarge",
            duration_seconds=7200,
        ),
        WorkloadResourceProfile(
            workload_id="test-inference-1",
            workload_type="inference",
            metrics=ResourceMetrics(cpu_utilization_percent=10.0, memory_utilization_percent=15.0),
            instance_type="ml.m5.large",
            duration_seconds=36000,
        ),
        WorkloadResourceProfile(
            workload_id="test-idle-1",
            workload_type="training",
            metrics=ResourceMetrics(
                cpu_utilization_percent=5.0,
                memory_utilization_percent=5.0,
                gpu_utilization_percent=0.0,
            ),
            instance_type="ml.p3.2xlarge",
            duration_seconds=18000,
        ),
        WorkloadResourceProfile(
            workload_id="test-gpu-1",
            workload_type="training",
            metrics=ResourceMetrics(
                cpu_utilization_percent=80.0,
                memory_utilization_percent=85.0,
                gpu_utilization_percent=90.0,
            ),
            instance_type="ml.p3.2xlarge",
            duration_seconds=7200,
        ),
    ]


def test_resource_analyzer_scores(resource_analyzer, mock_profiles):
    scores = resource_analyzer.analyze_utilization(mock_profiles[3])
    assert scores["cpu_score"] == 0.8
    assert scores["memory_score"] == 0.85
    assert scores["gpu_score"] == 0.9

    scores_no_gpu = resource_analyzer.analyze_utilization(mock_profiles[0])
    assert "gpu_score" not in scores_no_gpu
    assert scores_no_gpu["cpu_score"] == 0.5


def test_resource_analyzer_underutilized(resource_analyzer, mock_profiles):
    underutilized = resource_analyzer.identify_underutilized_resources(mock_profiles, threshold=0.2)
    assert len(underutilized) == 2
    assert underutilized[0].workload_id == "test-inference-1"
    assert underutilized[1].workload_id == "test-idle-1"


def test_cost_optimizer(cost_optimizer, mock_profiles):
    allocations = cost_optimizer.build_cost_allocation(mock_profiles)
    assert len(allocations) == 4

    batch_alloc = allocations["test-batch-1"]
    assert batch_alloc.duration_hours == 2.0
    assert batch_alloc.hourly_rate == 0.204
    assert batch_alloc.total_cost == 0.408

    total = cost_optimizer.calculate_total_cost(allocations)
    assert total > 0


def test_recommendation_engine_spot(recommendation_engine, cost_optimizer, mock_profiles):
    allocations = cost_optimizer.build_cost_allocation(mock_profiles)
    rec = recommendation_engine.add_spot_instance_recommendation(
        mock_profiles[0], allocations["test-batch-1"]
    )
    assert rec is not None
    assert rec.recommendation_type == "spot_instance"
    assert "save up to 70%" in rec.description


def test_recommendation_engine_auto_scaling(recommendation_engine, cost_optimizer, mock_profiles):
    allocations = cost_optimizer.build_cost_allocation(mock_profiles)
    rec = recommendation_engine.implement_auto_scaling_optimization(
        mock_profiles[1], allocations["test-inference-1"]
    )
    assert rec is not None
    assert rec.recommendation_type == "auto_scaling"
    assert "auto-scaling" in rec.description

    # High utilization inference should not get recommendation
    high_util_profile = WorkloadResourceProfile(
        workload_id="test-inference-2",
        workload_type="inference",
        metrics=ResourceMetrics(cpu_utilization_percent=80.0, memory_utilization_percent=80.0),
        instance_type="ml.m5.large",
        duration_seconds=36000,
    )
    high_util_alloc = cost_optimizer.calculate_workload_cost(high_util_profile)
    assert (
        recommendation_engine.implement_auto_scaling_optimization(
            high_util_profile, high_util_alloc
        )
        is None
    )


def test_recommendation_engine_right_sizing(recommendation_engine, cost_optimizer, mock_profiles):
    allocations = cost_optimizer.build_cost_allocation(mock_profiles)
    rec = recommendation_engine.add_savings_opportunity_identification(
        mock_profiles[2], allocations["test-idle-1"]
    )
    assert rec is not None
    assert rec.recommendation_type == "right_sizing"
    assert "Downsize instance" in rec.description


def test_generate_all_recommendations(recommendation_engine, cost_optimizer, mock_profiles):
    allocations = cost_optimizer.build_cost_allocation(mock_profiles)
    recs = recommendation_engine.generate_all_recommendations(mock_profiles, allocations)
    assert len(recs) == 3  # batch gets spot, inference gets auto_scale, idle gets right_sizing
