"""Tests for config hashing for reproducibility on registry entries."""

from __future__ import annotations

import pytest

from astroml.models.registry import ModelRegistry


class TestConfigHashing:
    def test_hash_is_deterministic(self):
        registry = ModelRegistry()
        training = {"epochs": 10, "lr": 0.001, "batch_size": 32}
        features = {"input_dim": 128, "normalize": True}

        h1 = registry.hash_config(training, features)
        h2 = registry.hash_config(training, features)
        assert h1 == h2

    def test_hash_changes_with_training_config(self):
        registry = ModelRegistry()
        h1 = registry.hash_config({"epochs": 10}, {"input_dim": 128})
        h2 = registry.hash_config({"epochs": 20}, {"input_dim": 128})
        assert h1 != h2

    def test_hash_changes_with_feature_config(self):
        registry = ModelRegistry()
        h1 = registry.hash_config({"epochs": 10}, {"input_dim": 128})
        h2 = registry.hash_config({"epochs": 10}, {"input_dim": 256})
        assert h1 != h2

    def test_hash_is_sha256_length(self):
        registry = ModelRegistry()
        h = registry.hash_config({}, {})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_maps_to_experiment_config(self):
        """Verifies that the same config always maps to the same hash."""
        registry = ModelRegistry()
        training_config = {
            "algorithm": "deep_svdd",
            "epochs": 50,
            "learning_rate": 0.0001,
            "batch_size": 64,
        }
        feature_config = {
            "feature_set": "temporal",
            "window_size": 100,
            "use_lookahead": True,
        }
        config_hash = registry.hash_config(training_config, feature_config)

        # Re-hashing identical configs must produce the same hash
        assert registry.hash_config(training_config, feature_config) == config_hash

        # Different training configs must produce different hashes
        modified_training = dict(training_config, epochs=100)
        assert registry.hash_config(modified_training, feature_config) != config_hash

    def test_hash_with_complex_nested_configs(self):
        registry = ModelRegistry()
        training = {
            "transformer": {"layers": 12, "heads": 8, "dim": 512},
            "optimizer": {"type": "adam", "beta1": 0.9, "beta2": 0.999},
        }
        features = {"pipeline": ["tokenize", "embed", "encode"], "normalize": True}
        h = registry.hash_config(training, features)
        assert len(h) == 64
        # Same structure, different key order must produce same hash
        training_reordered = {
            "optimizer": {"type": "adam", "beta1": 0.9, "beta2": 0.999},
            "transformer": {"layers": 12, "heads": 8, "dim": 512},
        }
        assert registry.hash_config(training_reordered, features) == h

    def test_hash_in_registry_version(self):
        """Verifies that config hash is stored on registered versions."""
        registry = ModelRegistry()
        training = {"epochs": 10}
        features = {"input_dim": 128}
        config_hash = registry.hash_config(training, features)

        registry.register("v1", "model-a", "/models/v1", config_hash=config_hash)
        assert registry.versions["v1"].config_hash == config_hash

    def test_different_configs_produce_unique_hashes(self):
        registry = ModelRegistry()
        configs = [
            ({"e": i}, {"d": i}) for i in range(100)
        ]
        hashes = set()
        for training, features in configs:
            h = registry.hash_config(training, features)
            assert h not in hashes
            hashes.add(h)
        assert len(hashes) == 100
