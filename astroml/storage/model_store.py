"""Dedicated model storage backend for saving, loading, and verifying ML model artifacts.

Provides serialization, checksum calculation, and storage management for PyTorch,
Scikit-learn, ONNX, and binary model formats with directory-based and cloud store integration.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from astroml.storage.artifact_store import ArtifactStore, LocalArtifactStore

logger = logging.getLogger(__name__)


def _collect_framework_metadata() -> dict[str, str]:
    """Collect runtime framework versions for enriched artifact metadata."""
    meta: dict[str, str] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch  # type: ignore
        meta["torch_version"] = torch.__version__
        meta["torch_cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        import sklearn  # type: ignore
        meta["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import numpy as np  # type: ignore
        meta["numpy_version"] = np.__version__
    except ImportError:
        pass
    return meta


class ModelStore:
    """Manages physical storage, serialization, and retrieval of model artifacts."""

    def __init__(
        self,
        base_dir: str | Path = "./artifacts/models",
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store or LocalArtifactStore(base_path=str(self.base_dir))

    def _get_version_dir(self, model_name: str, version: str) -> Path:
        sanitized_name = model_name.replace("/", "_").replace("\\", "_")
        sanitized_ver = version.replace("/", "_").replace("\\", "_")
        vdir = self.base_dir / sanitized_name / sanitized_ver
        vdir.mkdir(parents=True, exist_ok=True)
        return vdir

    def _compute_sha256(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def save_model(
        self,
        model_name: str,
        version: str,
        model_object: Any,
        filename: str = "model.pkl",
        metadata: dict[str, Any] | None = None,
        training_duration_secs: float | None = None,
        dataset_checksum: str | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        """Serialize and save an ML model object (pickle / torch / custom).

        Args:
            model_name: Logical model name for registry lookup.
            version: Version string (e.g. ``"v1.2.0"``).
            model_object: The model to persist.
            filename: Artifact filename inside the version directory.
            metadata: Arbitrary caller-supplied key/value metadata.
            training_duration_secs: Wall-clock training time in seconds.
            dataset_checksum: SHA-256 (or similar) digest of the training
                dataset, used for lineage tracking.
            output_schema: Mapping that describes the model's output tensor /
                DataFrame schema (e.g. ``{"labels": ["class_0", "class_1"]}``).
        """
        vdir = self._get_version_dir(model_name, version)
        target_file = vdir / filename

        # Serialization strategy
        if hasattr(model_object, "state_dict") and callable(model_object.state_dict):
            try:
                import torch

                torch.save(model_object.state_dict(), target_file)
            except Exception:
                with open(target_file, "wb") as f:
                    pickle.dump(model_object, f)
        elif isinstance(model_object, (bytes, bytearray)):
            with open(target_file, "wb") as f:
                f.write(model_object)
        elif isinstance(model_object, dict):
            with open(target_file, "wb") as f:
                pickle.dump(model_object, f)
        else:
            with open(target_file, "wb") as f:
                pickle.dump(model_object, f)

        checksum = self._compute_sha256(target_file)

        # Collect enriched metadata (issue #765)
        from astroml.storage.artifact_metadata import collect_metadata

        enriched = collect_metadata(
            model=model_object,
            dataset=dataset,
            training_duration=training_duration,
            training_config=training_config,
            sample_input=sample_input,
        )

        # Save metadata sidecar
        meta_payload: dict[str, Any] = {
            "model_name": model_name,
            "version": version,
            "filename": filename,
            "checksum_sha256": checksum,
            "size_bytes": target_file.stat().st_size,
            "framework": _collect_framework_metadata(),
            "custom_metadata": metadata or {},
            # Enriched fields (issue #765)
            "framework_version": enriched.framework_version,
            "torch_version": enriched.torch_version,
            "sklearn_version": enriched.sklearn_version,
            "dataset_checksum": enriched.dataset_checksum,
            "training_duration_seconds": enriched.training_duration_seconds,
            "output_schema": enriched.output_schema,
            "model_type": enriched.model_type,
            "git_commit": enriched.git_commit,
            "training_config": enriched.training_config or {},
            "registered_at": enriched.registered_at,
        }
        if training_duration_secs is not None:
            meta_payload["training_duration_secs"] = training_duration_secs
        if dataset_checksum is not None:
            meta_payload["dataset_checksum"] = dataset_checksum
        if output_schema is not None:
            meta_payload["output_schema"] = output_schema
        meta_file = vdir / f"{filename}.meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info("Saved model artifact: %s (SHA256: %s)", target_file, checksum[:8])
        return str(target_file)

    def load_model(
        self,
        model_name: str,
        version: str,
        filename: str = "model.pkl",
    ) -> Any:
        """Load and deserialize a model artifact."""
        vdir = self._get_version_dir(model_name, version)
        target_file = vdir / filename
        if not target_file.exists():
            raise FileNotFoundError(f"Model artifact not found at {target_file}")

        try:
            with open(target_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                import torch

                return torch.load(target_file, map_location="cpu")
            except Exception as e:
                with open(target_file, "rb") as f:
                    return f.read()

    def save_bytes(
        self,
        model_name: str,
        version: str,
        data: bytes,
        filename: str,
    ) -> str:
        """Save raw bytes as a model artifact file."""
        vdir = self._get_version_dir(model_name, version)
        target_file = vdir / filename
        with open(target_file, "wb") as f:
            f.write(data)
        checksum = self._compute_sha256(target_file)

        meta_file = vdir / f"{filename}.meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "version": version,
                    "filename": filename,
                    "checksum_sha256": checksum,
                    "size_bytes": len(data),
                },
                f,
                indent=2,
            )

        return str(target_file)

    def load_bytes(
        self,
        model_name: str,
        version: str,
        filename: str,
    ) -> bytes:
        """Read raw bytes from a model artifact file."""
        vdir = self._get_version_dir(model_name, version)
        target_file = vdir / filename
        if not target_file.exists():
            raise FileNotFoundError(f"Model artifact not found at {target_file}")
        with open(target_file, "rb") as f:
            return f.read()

    def exists(
        self,
        model_name: str,
        version: str,
        filename: str = "model.pkl",
    ) -> bool:
        """Check if a specific model artifact exists in storage."""
        vdir = self.base_dir / model_name.replace("/", "_") / version.replace("/", "_")
        return (vdir / filename).exists()

    def delete_version_artifacts(self, model_name: str, version: str) -> bool:
        """Delete all artifacts for a model version."""
        vdir = self.base_dir / model_name.replace("/", "_") / version.replace("/", "_")
        if vdir.exists():
            shutil.rmtree(vdir)
            logger.info("Deleted version artifacts at %s", vdir)
            return True
        return False

    def get_artifact_info(
        self,
        model_name: str,
        version: str,
        filename: str = "model.pkl",
    ) -> dict[str, Any]:
        """Get size, SHA256 checksum, and metadata for an artifact."""
        vdir = self._get_version_dir(model_name, version)
        target_file = vdir / filename
        if not target_file.exists():
            raise FileNotFoundError(f"Model artifact not found at {target_file}")

        checksum = self._compute_sha256(target_file)
        size = target_file.stat().st_size

        meta_file = vdir / f"{filename}.meta.json"
        sidecar: dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
            except Exception:
                pass

        info: dict[str, Any] = {
            "model_name": model_name,
            "version": version,
            "filename": filename,
            "path": str(target_file),
            "size_bytes": size,
            "checksum_sha256": checksum,
            "custom_metadata": sidecar.get("custom_metadata", {}),
            "framework": sidecar.get("framework", {}),
        }
        for enriched_key in ("training_duration_secs", "dataset_checksum", "output_schema"):
            if enriched_key in sidecar:
                info[enriched_key] = sidecar[enriched_key]
        return info

    def list_version_artifacts(self, model_name: str, version: str) -> list[str]:
        """List all artifact filenames for a model version."""
        vdir = self.base_dir / model_name.replace("/", "_") / version.replace("/", "_")
        if not vdir.exists():
            return []
        return [f.name for f in vdir.iterdir() if f.is_file() and not f.name.endswith(".meta.json")]

    def verify_checksum(
        self,
        model_name: str,
        version: str,
        filename: str,
        expected_checksum: str,
    ) -> bool:
        """Verify that stored artifact matches expected SHA256 checksum."""
        info = self.get_artifact_info(model_name, version, filename)
        return info["checksum_sha256"].lower() == expected_checksum.lower()
