"""Automated data labeling and annotation pipeline (issue #624).

Provides active learning query strategies, weak supervision via labeling
functions, human-in-the-loop review, and pipeline orchestration.

Components:
- ActiveLearner: Active learning with uncertainty/diversity/hybrid strategies
- WeakSupervisionModel: Programmatic labeling with LFs and generative models
- ReviewQueue: Prioritised human review with conflict resolution
- LabelingPipeline: End-to-end labeling orchestration
"""

from __future__ import annotations

from .active_learning import (
    ActiveLearner,
    ActiveLearningResult,
    DiversitySampling,
    EntropySampling,
    HybridStrategy,
    MarginSampling,
    QueryStrategy,
    SampleScore,
    UncertaintySampling,
)
from .review_queue import ConflictResolver, LabelQualityMetrics, ReviewQueue
from .strategies import (
    ActiveWeakStrategy,
    BatchLabelingStrategy,
    LabelingDashboard,
    LabelingPipeline,
    LabelingResult,
    LabelingStrategy,
)
from .weak_supervision import (
    LabelingFunction,
    MajorityVoter,
    WeakSupervisionModel,
    create_binary_lfs,
)

__all__ = [
    # Active learning
    "ActiveLearner",
    "ActiveLearningResult",
    "DiversitySampling",
    "EntropySampling",
    "HybridStrategy",
    "MarginSampling",
    "QueryStrategy",
    "SampleScore",
    "UncertaintySampling",
    # Weak supervision
    "LabelingFunction",
    "MajorityVoter",
    "WeakSupervisionModel",
    "create_binary_lfs",
    # Review
    "ConflictResolver",
    "LabelQualityMetrics",
    "ReviewQueue",
    # Strategies / pipeline
    "ActiveWeakStrategy",
    "BatchLabelingStrategy",
    "LabelingDashboard",
    "LabelingPipeline",
    "LabelingResult",
    "LabelingStrategy",
]