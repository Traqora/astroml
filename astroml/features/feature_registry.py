"""Feature Registry for managing feature metadata, definitions, types, and versions.

Provides centralized registration, discovery, validation, and lineage tracking
for all feature computers in the AstroML system.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Supported feature data types."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    VECTOR = "vector"
    TIME_SERIES = "time_series"


class FeatureStatus(Enum):
    """Feature lifecycle status."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class FeatureDefinition:
    """Definition and metadata for a registered feature."""

    name: str
    description: str
    feature_type: FeatureType
    computation_function: Callable[..., Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    owner: str = "system"
    status: FeatureStatus = FeatureStatus.DEVELOPMENT
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_id(self) -> str:
        """Unique identifier formatted with version."""
        return f"{self.name}_v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize feature definition to dictionary."""
        return {
            "name": self.name,
            "feature_id": self.feature_id,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "parameters": self.parameters,
            "tags": self.tags,
            "owner": self.owner,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDefinition:
        """Deserialize feature definition from dictionary."""
        d = data.copy()
        d.pop("feature_id", None)
        if isinstance(d.get("feature_type"), str):
            d["feature_type"] = FeatureType(d["feature_type"])
        if isinstance(d.get("status"), str):
            d["status"] = FeatureStatus(d["status"])
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        if isinstance(d.get("updated_at"), str):
            d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return cls(**d)


@runtime_checkable
class FeatureComputer(Protocol):
    """Protocol for feature computation functions."""

    def __call__(
        self,
        data: Any,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> Any:
        """Compute feature values."""
        ...


class FeatureRegistryService:
    """Registry service managing definitions and feature computation hooks."""

    def __init__(self) -> None:
        """Initialize feature registry."""
        self._definitions: dict[str, FeatureDefinition] = {}
        self._computers: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        computer: Callable[..., Any] | None = None,
        description: str = "",
        feature_type: FeatureType = FeatureType.NUMERIC,
        tags: list[str] | None = None,
        owner: str = "system",
        parameters: dict[str, Any] | None = None,
        version: int = 1,
        status: FeatureStatus = FeatureStatus.PRODUCTION,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureDefinition:
        """Register a feature definition and optional computer."""
        feat_def = FeatureDefinition(
            name=name,
            description=description,
            feature_type=feature_type,
            computation_function=computer,
            parameters=parameters or {},
            tags=tags or [],
            owner=owner,
            status=status,
            version=version,
            metadata=metadata or {},
        )
        self._definitions[name] = feat_def
        self._definitions[feat_def.feature_id] = feat_def
        if computer is not None:
            self._computers[name] = computer
            self._computers[feat_def.feature_id] = computer
        logger.info("Registered feature: %s (v%d)", name, version)
        return feat_def

    def get_definition(self, name_or_id: str) -> FeatureDefinition | None:
        """Get feature definition by name or feature_id."""
        return self._definitions.get(name_or_id)

    def get_computer(self, name_or_id: str) -> Callable[..., Any] | None:
        """Get feature computation function."""
        return self._computers.get(name_or_id)

    def list_definitions(
        self,
        status: FeatureStatus | None = None,
        tags: Sequence[str] | None = None,
        owner: str | None = None,
    ) -> list[FeatureDefinition]:
        """List registered feature definitions matching filter criteria."""
        unique_defs = {d.name: d for d in self._definitions.values()}.values()
        results = []
        for d in unique_defs:
            if status is not None and d.status != status:
                continue
            if owner is not None and d.owner != owner:
                continue
            if tags is not None and not all(t in d.tags for t in tags):
                continue
            results.append(d)
        return results


def create_feature_registry() -> FeatureRegistryService:
    """Factory function for FeatureRegistryService."""
    return FeatureRegistryService()
