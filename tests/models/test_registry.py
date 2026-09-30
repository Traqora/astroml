"""Tests for the model registry with import/export and config hashing."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from astroml.models.registry import ExportBundle, ModelRegistry, VersionConfig


class TestModelRegistry:
    def test_register_and_retrieve(self):
        registry = ModelRegistry()
        config = registry.register("v1", "fraud-model", "/models/v1")
        assert config.version_id == "v1"
        assert registry.versions["v1"].model_name == "fraud-model"

    def test_register_multiple_versions(self):
        registry = ModelRegistry()
        registry.register("v1", "model-a", "/models/v1")
        registry.register("v2", "model-a", "/models/v2")
        assert len(registry.versions) == 2

    def test_export_version(self):
        registry = ModelRegistry()
        registry.register("v1", "fraud-model", "/models/v1", metrics={"f1": 0.95})
        bundle = registry.export_version("v1")
        assert isinstance(bundle, ExportBundle)
        assert bundle.version_id == "v1"
        assert bundle.model_name == "fraud-model"
        assert bundle.metrics == {"f1": 0.95}

    def test_export_version_not_found(self):
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="not found"):
            registry.export_version("nonexistent")

    def test_import_version(self):
        registry = ModelRegistry()
        registry.register("v1", "fraud-model", "/models/v1", metrics={"f1": 0.95})
        bundle = registry.export_version("v1")

        new_registry = ModelRegistry()
        config = new_registry.import_version(bundle)
        assert config.version_id == "v1"
        assert config.activation_state == "inactive"
        assert new_registry.versions["v1"].metrics == {"f1": 0.95}

    def test_import_version_duplicate(self):
        registry = ModelRegistry()
        registry.register("v1", "fraud-model", "/models/v1")
        bundle = ExportBundle(
            version_id="v1", model_name="fraud-model", artifact_path="/models/v1",
            metadata={}, metrics={}, lineage={}, activation_state="inactive",
            created_at="2024-01-01T00:00:00Z", config_hash="",
        )
        with pytest.raises(ValueError, match="already exists"):
            new_registry.import_version(bundle)

    def test_activate_deactivate(self):
        registry = ModelRegistry()
        registry.register("v1", "fraud-model", "/models/v1")
        registry.activate("v1")
        assert registry.versions["v1"].activation_state == "active"
        assert registry.get_activated().version_id == "v1"

        registry.deactivate("v1")
        assert registry.versions["v1"].activation_state == "inactive"
        assert registry.get_activated() is None

    def test_hash_config_deterministic(self):
        registry = ModelRegistry()
        training_config = {"epochs": 10, "lr": 0.001}
        feature_config = {"input_dim": 128, "normalize": True}

        hash1 = registry.hash_config(training_config, feature_config)
        hash2 = registry.hash_config(training_config, feature_config)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_hash_config_different(self):
        registry = ModelRegistry()
        h1 = registry.hash_config({"epochs": 10}, {"input_dim": 128})
        h2 = registry.hash_config({"epochs": 20}, {"input_dim": 128})
        assert h1 != h2

    def test_serialize_deserialize(self):
        registry = ModelRegistry()
        registry.register("v1", "model-a", "/models/v1")
        registry.activate("v1")
        registry.register("v2", "model-b", "/models/v2")

        data = registry.to_dict()
        restored = ModelRegistry.from_dict(data)

        assert len(restored.versions) == 2
        assert restored.get_activated().version_id == "v1"
        assert restored.versions["v2"].model_name == "model-b"

    def test_versioning_lineage_preserved(self):
        registry = ModelRegistry()
        config = registry.register(
            "v1", "fraud-model", "/models/v1",
            lineage={"parent": "v0", "experiment": "exp-42"},
            metrics={"f1": 0.95, "latency_ms": 12.5},
        )
        assert config.lineage == {"parent": "v0", "experiment": "exp-42"}
        assert config.metrics == {"f1": 0.95, "latency_ms": 12.5}


class TestExportImportRoundTrip:
    def test_full_round_trip(self):
        original = ModelRegistry()
        original.register(
            "v1", "fraud-model", "/models/v1",
            metadata={"author": "team-a", "description": "Fraud detection"},
            metrics={"f1": 0.95, "accuracy": 0.92},
            lineage={"base_model": "v0", "dataset": "fraud-v3"},
            activation_state="active",
        )

        bundle = original.export_version("v1")
        new_registry = ModelRegistry()
        imported = new_registry.import_version(bundle)

        assert imported.version_id == "v1"
        assert imported.activation_state == "active"
        assert imported.metrics["f1"] == 0.95
        assert imported.lineage["dataset"] == "fraud-v3"

    def test_json_serializable_bundle(self):
        bundle = ExportBundle(
            version_id="v1", model_name="model-a", artifact_path="/models/v1",
            metadata={}, metrics={"f1": 0.9}, lineage={},
            activation_state="active", created_at="2024-01-01T00:00:00Z",
            config_hash="abc123",
        )
        # Verify the bundle can be serialized to dict and back
        data = {
            "version_id": bundle.version_id,
            "model_name": bundle.model_name,
            "artifact_path": bundle.artifact_path,
            "metadata": bundle.metadata,
            "metrics": bundle.metrics,
            "lineage": bundle.lineage,
            "activation_state": bundle.activation_state,
            "created_at": bundle.created_at,
            "config_hash": bundle.config_hash,
        }
        assert json.loads(json.dumps(data)) is not None
