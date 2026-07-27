"""LLM feature definitions for the Feature Store.

Defines feature types and metadata specific to LLM-generated features
including embeddings, scores, explanations, and meta attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMFeatureCategory(Enum):
    """Categories of LLM-generated features."""
    EMBEDDING = "embedding"
    SCORE = "score"
    EXPLANATION = "explanation"
    META = "meta"


class EmbeddingType(Enum):
    """Types of embeddings supported."""
    TRANSACTION_DESCRIPTION = "transaction_description"
    ACCOUNT_BEHAVIOR = "account_behavior"
    ALERT_DESCRIPTION = "alert_description"


class ScoreType(Enum):
    """Types of LLM-generated scores."""
    FRAUD_PROBABILITY = "fraud_probability"
    EXPLANATION_CONFIDENCE = "explanation_confidence"
    UNCERTAINTY_ESTIMATE = "uncertainty_estimate"


@dataclass
class LLMFeatureMeta:
    """Metadata for an LLM-generated feature."""
    prompt_version: str
    model_name: str
    provider: str
    latency_ms: float
    tokens_used: int
    cost_usd: float
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LLMFeatureDefinition:
    """Definition of an LLM-generated feature."""
    name: str
    category: LLMFeatureCategory
    description: str
    dimension: Optional[int] = None
    embedding_type: Optional[EmbeddingType] = None
    score_type: Optional[ScoreType] = None
    tags: List[str] = field(default_factory=list)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "dimension": self.dimension,
            "embedding_type": self.embedding_type.value if self.embedding_type else None,
            "score_type": self.score_type.value if self.score_type else None,
            "tags": self.tags,
            "version": self.version,
            "metadata": self.metadata,
        }
