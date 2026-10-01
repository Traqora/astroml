"""Model registry with import/export and config hashing for reproducibility.

Provides:
- export_version: serializes a model version (artifact + metadata + metrics)
  for backup and sharing across registry instances.
- import_version: deserializes an exported version into another registry
  instance, preserving versioning, lineage, and activation state.
- hash_config: computes a content-addressable hash of training and
  feature configs so any artifact can be mapped back to the exact
  experiment configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class VersionConfig:
    """Configuration for a registered model version."""

    version_id: str
    model_name: str
    artifact_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    activation_state: str = "inactive"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""


@dataclass
class ExportBundle:
    """A serializable bundle representing a full model version export."""

    version_id: str
    model_name: str
    artifact_path: str
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    lineage: dict[str, Any]
    activation_state: str
    created_at: str
    config_hash: str


class ModelRegistry:
    """Registry for managing model versions with import/export and config hashing."""

    def __init__(self) -> None:
        self.versions: dict[str, VersionConfig] = {}
        self._activated_version: str | None = None

    def register(
        self,
        version_id: str,
        model_name: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        activation_state: str = "inactive",
        config_hash: str = "",
    ) -> VersionConfig:
        """Register a new model version in the registry."""
        config = VersionConfig(
            version_id=version_id,
            model_name=model_name,
            artifact_path=artifact_path,
            metadata=metadata or {},
            metrics=metrics or {},
            lineage=lineage or {},
            activation_state=activation_state,
            config_hash=config_hash,
        )
        self.versions[version_id] = config
        return config

    def export_version(self, version_id: str) -> ExportBundle:
        """Export a model version as a portable bundle (artifact + metadata + metrics).

        Returns an ExportBundle that can be serialized to JSON and imported
        into another registry instance.
        """
        config = self.versions.get(version_id)
        if config is None:
            raise ValueError(f"Version {version_id} not found in registry")

        return ExportBundle(
            version_id=config.version_id,
            model_name=config.model_name,
            artifact_path=config.artifact_path,
            metadata=config.metadata,
            metrics=config.metrics,
            lineage=config.lineage,
            activation_state=config.activation_state,
            created_at=config.created_at,
            config_hash=config.config_hash,
        )

    def import_version(self, bundle: ExportBundle) -> VersionConfig:
        """Import a model version from an exported bundle into this registry.

        Preserves versioning, lineage, and activation state. The version_id
        must not already exist in the registry (to avoid overwriting).
        """
        if bundle.version_id in self.versions:
            raise ValueError(
                f"Version {bundle.version_id} already exists; use a different version_id"
            )

        config = VersionConfig(
            version_id=bundle.version_id,
            model_name=bundle.model_name,
            artifact_path=bundle.artifact_path,
            metadata=bundle.metadata,
            metrics=bundle.metrics,
            lineage=bundle.lineage,
            activation_state=bundle.activation_state,
            created_at=bundle.created_at,
            config_hash=bundle.config_hash,
        )
        self.versions[bundle.version_id] = config
        return config

    def hash_config(self, training_config: dict[str, Any], feature_config: dict[str, Any]) -> str:
        """Hash training and feature configs into a content-addressable hash.

        Returns a SHA-256 hex digest that maps any artifact back to the
        exact experiment configuration. The hash is deterministic: identical
        configs always produce the same digest.
        """
        payload = {
            "training": training_config,
            "feature": feature_config,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def activate(self, version_id: str) -> None:
        """Activate a model version for serving."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        self._activated_version = version_id
        self.versions[version_id].activation_state = "active"

    def deactivate(self, version_id: str) -> None:
        """Deactivate a model version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        self.versions[version_id].activation_state = "inactive"
        if self._activated_version == version_id:
            self._activated_version = None

    def get_activated(self) -> VersionConfig | None:
        """Return the currently activated version, or None."""
        if self._activated_version is None:
            return None
        return self.versions.get(self._activated_version)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry to a dict for JSON export."""
        return {
            "versions": {
                vid: {
                    "version_id": c.version_id,
                    "model_name": c.model_name,
                    "artifact_path": c.artifact_path,
                    "metadata": c.metadata,
                    "metrics": c.metrics,
                    "lineage": c.lineage,
                    "activation_state": c.activation_state,
                    "created_at": c.created_at,
                    "config_hash": c.config_hash,
                }
                for vid, c in self.versions.items()
            },
            "activated_version": self._activated_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRegistry:
        """Deserialize a registry from a dict created by to_dict."""
        registry = cls()
        for vid, vdata in data.get("versions", {}).items():
            config = VersionConfig(
                version_id=vdata["version_id"],
                model_name=vdata["model_name"],
                artifact_path=vdata["artifact_path"],
                metadata=vdata["metadata"],
                metrics=vdata["metrics"],
                lineage=vdata["lineage"],
                activation_state=vdata["activation_state"],
                created_at=vdata["created_at"],
                config_hash=vdata["config_hash"],
            )
            registry.versions[vid] = config
        registry._activated_version = data.get("activated_version")
        return registry
