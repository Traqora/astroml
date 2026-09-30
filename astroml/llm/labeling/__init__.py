"""LLM-based data labeling system for issue #475.

Provides automated data labeling with LLM validation for ML training data.

Components:
- Labeler: Core labeling logic
- Schemas: Label schema definitions
- Validators: Label validation
- Consensus: Multi-LLM consensus
- Human: Human-in-the-loop integration
"""

from __future__ import annotations

from .consensus import ConsensusLabeler, ConsensusResult
from .human import HumanReviewQueue, ReviewTask
from .labeler import DataLabeler, LabelResult
from .schemas import LabelDefinition, LabelSchema, LabelType
from .validators import LabelValidator, ValidationRule

__all__ = [
    "DataLabeler",
    "LabelResult",
    "LabelSchema",
    "LabelType",
    "LabelDefinition",
    "LabelValidator",
    "ValidationRule",
    "ConsensusLabeler",
    "ConsensusResult",
    "HumanReviewQueue",
    "ReviewTask",
]
