"""Tracking utilities (metrics, usage, experiment tracking, etc)."""

# ---------------------------------------------------------------------------
# Imports & Package Exports
# ---------------------------------------------------------------------------
from .ab_testing import ABTestingFramework
from .mlflow_tracker import MLflowTracker
from .model_registry import ModelRegistry
from .llm_usage_tracker import (
    LLMUsage,
    LLMPrices,
    LLMUsageTracker,
    default_llm_usage_tracker,
)

# Combined clean export list (Fixing the duplicate __all__ bug from your branch)
__all__ = [
    "ABTestingFramework",
    "MLflowTracker",
    "ModelRegistry",
    "LLMUsage",
    "LLMPrices",
    "LLMUsageTracker",
    "default_llm_usage_tracker",
]


# ---------------------------------------------------------------------------
# A/B Testing Framework
# ---------------------------------------------------------------------------

class Experiment(Base):
    """A/B test experiment for comparing models or prompts."""
    __tablename__ = "experiments"
    # ... keep full definition

class Variant(Base):
    """A variant in an A/B test experiment."""
    __tablename__ = "variants"
    # ... keep full definition

class ExperimentResult(Base):
    """Individual result from an A/B test experiment."""
    __tablename__ = "experiment_results"
    # ... keep full definition


# ---------------------------------------------------------------------------
# Golden Dataset Framework
# ---------------------------------------------------------------------------

class GoldenDataset(Base):
    """Golden dataset for model evaluation and benchmarking."""
    __tablename__ = "golden_datasets"
    # ... keep full definition

class GoldenDatasetEntry(Base):
    """Individual entry in a golden dataset with ground truth labels."""
    __tablename__ = "golden_dataset_entries"
    # ... keep full definition


# ---------------------------------------------------------------------------
# Ledger Processing
# ---------------------------------------------------------------------------

class ProcessedLedger(Base):
    """Tracking table for processed ledgers during backfill to ensure idempotency."""
    __tablename__ = "processed_ledgers"
    # ... keep full definition