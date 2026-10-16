"""Tracking utilities (metrics, usage, experiment tracking, data lineage, etc)."""

# ---------------------------------------------------------------------------
# Imports & Package Exports
# ---------------------------------------------------------------------------
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ABTestingFramework",
    "Experiment",
    "ExperimentDashboard",
    "ExperimentReport",
    "MLflowTracker",
    "EpochRecord",
    "ModelRegistry",
    "ModelStage",
    "TrainingReport",
    "DeploymentEnvironment",
    "SemanticVersion",
    "ModelMetadata",
    "ModelFramework",
    "TaskType",
    "LLMUsage",
    "LLMPrices",
    "LLMUsageTracker",
    "default_llm_usage_tracker",
    "RunComparator",
    "RunMetrics",
    "ChartData",
    "ExperimentVisualizer",
    "DataLineageTracker",
    "ProvenanceTracker",
    "LineageVisualizer",
    "MetadataStore",
]

_LAZY: dict[str, tuple[str, str]] = {
    "ABTestingFramework": ("astroml.tracking.ab_testing", "ABTestingFramework"),
    "Experiment": ("astroml.tracking.experiment_dashboard", "Experiment"),
    "ExperimentDashboard": ("astroml.tracking.experiment_dashboard", "ExperimentDashboard"),
    "ExperimentReport": ("astroml.tracking.experiment_dashboard", "ExperimentReport"),
    "MLflowTracker": ("astroml.tracking.mlflow_tracker", "MLflowTracker"),
    "EpochRecord": ("astroml.tracking.training_report", "EpochRecord"),
    "ModelRegistry": ("astroml.tracking.model_registry", "ModelRegistry"),
    "ModelStage": ("astroml.tracking.model_registry", "ModelStage"),
    "TrainingReport": ("astroml.tracking.training_report", "TrainingReport"),
    "DeploymentEnvironment": ("astroml.tracking.model_registry", "DeploymentEnvironment"),
    "SemanticVersion": ("astroml.tracking.model_registry", "SemanticVersion"),
    "ModelMetadata": ("astroml.tracking.metadata", "ModelMetadata"),
    "ModelFramework": ("astroml.tracking.metadata", "ModelFramework"),
    "TaskType": ("astroml.tracking.metadata", "TaskType"),
    "LLMUsage": ("astroml.tracking.llm_usage_tracker", "LLMUsage"),
    "LLMPrices": ("astroml.tracking.llm_usage_tracker", "LLMPrices"),
    "LLMUsageTracker": ("astroml.tracking.llm_usage_tracker", "LLMUsageTracker"),
    "default_llm_usage_tracker": (
        "astroml.tracking.llm_usage_tracker",
        "default_llm_usage_tracker",
    ),
    "RunComparator": ("astroml.tracking.run_comparator", "RunComparator"),
    "RunMetrics": ("astroml.tracking.run_comparator", "RunMetrics"),
    "ChartData": ("astroml.tracking.visualizations", "ChartData"),
    "ExperimentVisualizer": ("astroml.tracking.visualizations", "ExperimentVisualizer"),
    "DataLineageTracker": ("astroml.tracking.lineage.data_lineage", "DataLineageTracker"),
    "ProvenanceTracker": ("astroml.tracking.lineage.provenance", "ProvenanceTracker"),
    "LineageVisualizer": ("astroml.tracking.lineage.visualizer", "LineageVisualizer"),
    "MetadataStore": ("astroml.tracking.lineage.metadata_store", "MetadataStore"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
