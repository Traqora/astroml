"""Tracking utilities (metrics, usage, experiment tracking, data lineage, etc)."""

# ---------------------------------------------------------------------------
# Imports & Package Exports
# ---------------------------------------------------------------------------
from __future__ import annotations

from importlib import import_module

from .ab_testing import ABTestingFramework
from .experiment_dashboard import (
    Experiment,
    ExperimentDashboard,
    ExperimentReport,
)
from .llm_usage_tracker import (
    LLMPrices,
    LLMUsage,
    LLMUsageTracker,
    default_llm_usage_tracker,
)
from .metadata import ModelFramework, ModelMetadata, TaskType
from .mlflow_tracker import MLflowTracker
from .model_registry import (
    DeploymentEnvironment,
    ModelRegistry,
    ModelStage,
    SemanticVersion,
)
from .run_comparator import RunComparator, RunMetrics
from .training_report import EpochRecord, TrainingReport
from .visualizations import ChartData, ExperimentVisualizer

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
    "TrainingLineage",
    "ModelLineage",
]

_LAZY: dict[str, tuple[str, str]] = {
    "DataLineageTracker": ("astroml.tracking.lineage.data_lineage", "DataLineageTracker"),
    "ProvenanceTracker": ("astroml.tracking.lineage.provenance", "ProvenanceTracker"),
    "LineageVisualizer": ("astroml.tracking.lineage.visualizer", "LineageVisualizer"),
    "MetadataStore": ("astroml.tracking.lineage.metadata_store", "MetadataStore"),
    "TrainingLineage": ("astroml.tracking.lineage.data_lineage", "TrainingLineage"),
    "ModelLineage": ("astroml.tracking.lineage.data_lineage", "ModelLineage"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

