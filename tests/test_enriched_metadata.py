"""Tests for enriched artifact metadata on model registry entries — issue #765."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from astroml.storage.model_store import ModelStore, _collect_framework_metadata


class TestCollectFrameworkMetadata:
    def test_returns_dict_with_python_version(self):
        meta = _collect_framework_metadata()
        assert "python_version" in meta
        assert meta["python_version"]

    def test_returns_platform(self):
        meta = _collect_framework_metadata()
        assert "platform" in meta


class TestEnrichedMetadataSidecar:
    def _make_store(self, tmp_path: Path) -> ModelStore:
        return ModelStore(base_dir=str(tmp_path / "models"))

    def test_basic_save_writes_sidecar(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save_model("my_model", "v1", {"weights": [1, 2, 3]})

        sidecar_path = (
            tmp_path / "models" / "my_model" / "v1" / "model.pkl.meta.json"
        )
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert "framework" in sidecar
        assert "python_version" in sidecar["framework"]

    def test_training_duration_stored(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save_model(
            "m", "v1", b"bytes",
            filename="model.bin",
            training_duration_secs=42.5,
        )
        sidecar_path = tmp_path / "models" / "m" / "v1" / "model.bin.meta.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["training_duration_secs"] == pytest.approx(42.5)

    def test_dataset_checksum_stored(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save_model(
            "m", "v1", {"w": 1},
            dataset_checksum="abc123",
        )
        sidecar_path = tmp_path / "models" / "m" / "v1" / "model.pkl.meta.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["dataset_checksum"] == "abc123"

    def test_output_schema_stored(self, tmp_path):
        store = self._make_store(tmp_path)
        schema = {"labels": ["fraud", "legit"], "threshold": 0.5}
        store.save_model("m", "v1", {"w": 1}, output_schema=schema)
        sidecar_path = tmp_path / "models" / "m" / "v1" / "model.pkl.meta.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["output_schema"] == schema

    def test_optional_fields_absent_when_not_provided(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save_model("m", "v1", {"w": 1})
        sidecar_path = tmp_path / "models" / "m" / "v1" / "model.pkl.meta.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert "training_duration_secs" not in sidecar
        assert "dataset_checksum" not in sidecar
        assert "output_schema" not in sidecar

    def test_get_artifact_info_surfaces_enriched_fields(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save_model(
            "m", "v1", {"w": 1},
            training_duration_secs=10.0,
            dataset_checksum="deadbeef",
            output_schema={"out": "float32"},
        )
        info = store.get_artifact_info("m", "v1")
        assert info["training_duration_secs"] == pytest.approx(10.0)
        assert info["dataset_checksum"] == "deadbeef"
        assert info["output_schema"] == {"out": "float32"}
        assert "framework" in info
        assert "python_version" in info["framework"]

    def test_custom_metadata_still_accessible(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save_model(
            "m", "v1", {"w": 1},
            metadata={"experiment_id": "exp-42"},
        )
        info = store.get_artifact_info("m", "v1")
        assert info["custom_metadata"]["experiment_id"] == "exp-42"
