"""Tests for ModelStore artifact persistence, retrieval, and checksum verification."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from astroml.storage.model_store import ModelStore


class DummyLinearModel:
    def __init__(self, coef: float):
        self.coef = coef

    def predict(self, x: float) -> float:
        return self.coef * x


class TestModelStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ModelStore(base_dir=tmp_path / "models")

    def test_save_and_load_model_object(self, store):
        model = DummyLinearModel(coef=3.14)
        path = store.save_model("gnn_classifier", "1.0.0", model)
        assert os.path.exists(path)
        assert store.exists("gnn_classifier", "1.0.0")

        loaded = store.load_model("gnn_classifier", "1.0.0")
        assert isinstance(loaded, DummyLinearModel)
        assert loaded.predict(2.0) == 6.28

    def test_save_and_load_bytes(self, store):
        raw_bytes = b"model_weights_raw_buffer_12345"
        path = store.save_bytes("anomaly_detector", "0.1.0", raw_bytes, "weights.bin")
        assert os.path.exists(path)

        loaded_bytes = store.load_bytes("anomaly_detector", "0.1.0", "weights.bin")
        assert loaded_bytes == raw_bytes

    def test_checksum_verification(self, store):
        model = {"layer1": [1, 2, 3], "layer2": [4, 5, 6]}
        store.save_model("tree_model", "2.0.0", model)

        info = store.get_artifact_info("tree_model", "2.0.0")
        assert "checksum_sha256" in info
        assert len(info["checksum_sha256"]) == 64
        assert store.verify_checksum("tree_model", "2.0.0", "model.pkl", info["checksum_sha256"])
        assert not store.verify_checksum("tree_model", "2.0.0", "model.pkl", "invalid_hash_1234")

    def test_delete_version_artifacts(self, store):
        store.save_model("to_delete", "1.0.0", {"a": 1})
        assert store.exists("to_delete", "1.0.0")

        deleted = store.delete_version_artifacts("to_delete", "1.0.0")
        assert deleted
        assert not store.exists("to_delete", "1.0.0")
