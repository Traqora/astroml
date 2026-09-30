"""Label schema definitions for LLM-based data labeling (issue #475)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LabelType(Enum):
    """Types of labels that can be generated."""

    CLASSIFICATION = "classification"  # Single class label
    MULTI_LABEL = "multi_label"  # Multiple labels
    ENTITY = "entity"  # Named entity recognition
    SENTIMENT = "sentiment"  # Sentiment analysis
    MATCH = "match"  # Entity resolution
    CUSTOM = "custom"  # Custom label type


@dataclass
class LabelDefinition:
    """Definition of a label type.

    Attributes:
        name: Label name
        label_type: Type of label
        description: Label description
        allowed_values: Allowed label values (for classification)
        confidence_threshold: Minimum confidence for auto-accept
        requires_human_review: Whether low-confidence labels need review
    """

    name: str
    label_type: LabelType
    description: str
    allowed_values: Optional[List[str]] = None
    confidence_threshold: float = 0.85
    requires_human_review: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "label_type": self.label_type.value,
            "description": self.description,
            "allowed_values": self.allowed_values,
            "confidence_threshold": self.confidence_threshold,
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class Label:
    """A single label.

    Attributes:
        id: Unique identifier
        label_name: Name of the label
        value: Label value
        confidence: Confidence score (0-1)
        source: Source of the label (llm, human, consensus)
        metadata: Additional metadata
        created_at: When the label was created
    """

    id: str
    label_name: str
    value: Any
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label_name": self.label_name,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LabelSchema:
    """Schema for labeling tasks.

    Attributes:
        name: Schema name
        description: Schema description
        label_definitions: List of label definitions
        version: Schema version
        created_at: When the schema was created
    """

    name: str
    description: str
    label_definitions: List[LabelDefinition] = field(default_factory=list)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_label_definition(self, definition: LabelDefinition) -> None:
        """Add a label definition to the schema.

        Args:
            definition: Label definition to add
        """
        self.label_definitions.append(definition)

    def get_definition(self, name: str) -> Optional[LabelDefinition]:
        """Get a label definition by name.

        Args:
            name: Label name

        Returns:
            LabelDefinition if found
        """
        for defn in self.label_definitions:
            if defn.name == name:
                return defn
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "label_definitions": [d.to_dict() for d in self.label_definitions],
            "created_at": self.created_at.isoformat(),
        }


# Predefined schemas for common tasks

FRAUD_CLASSIFICATION_SCHEMA = LabelSchema(
    name="fraud_classification",
    description="Classification of transactions as fraudulent, suspicious, or legitimate",
    label_definitions=[
        LabelDefinition(
            name="fraud_status",
            label_type=LabelType.CLASSIFICATION,
            description="Fraud classification status",
            allowed_values=["fraudulent", "suspicious", "legitimate"],
            confidence_threshold=0.85,
            requires_human_review=True,
        ),
    ],
)

ALERT_CATEGORIZATION_SCHEMA = LabelSchema(
    name="alert_categorization",
    description="Categorization of fraud alerts by pattern type and severity",
    label_definitions=[
        LabelDefinition(
            name="pattern_type",
            label_type=LabelType.CLASSIFICATION,
            description="Type of fraud pattern",
            allowed_values=["circular", "layering", "structuring", "unknown"],
            confidence_threshold=0.80,
            requires_human_review=True,
        ),
        LabelDefinition(
            name="severity",
            label_type=LabelType.CLASSIFICATION,
            description="Alert severity level",
            allowed_values=["high", "medium", "low"],
            confidence_threshold=0.85,
            requires_human_review=False,
        ),
    ],
)

ENTITY_RESOLUTION_SCHEMA = LabelSchema(
    name="entity_resolution",
    description="Matching accounts across different data sources",
    label_definitions=[
        LabelDefinition(
            name="match_status",
            label_type=LabelType.MATCH,
            description="Whether accounts represent the same entity",
            allowed_values=["match", "no_match", "uncertain"],
            confidence_threshold=0.90,
            requires_human_review=True,
        ),
    ],
)

SENTIMENT_SCHEMA = LabelSchema(
    name="sentiment_analysis",
    description="Sentiment analysis of user feedback",
    label_definitions=[
        LabelDefinition(
            name="sentiment",
            label_type=LabelType.SENTIMENT,
            description="Sentiment category",
            allowed_values=["positive", "neutral", "negative"],
            confidence_threshold=0.80,
            requires_human_review=False,
        ),
    ],
)
