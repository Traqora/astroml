"""Enriched artifact metadata for model registry entries — issue #765.

Attaches useful metadata on registration: framework and torch versions,
dataset checksum, training duration, and output schema.  Provides
backward-compatible migration: existing entries without the new fields
continue to work, and a ``migrate_registry_entry`` helper can enrich
legacy entries on demand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Enriched metadata attached to every model registry entry.

    All fields are optional to maintain backward compatibility with
    existing registry entries that lack this information.

    Attributes
    ----------
    framework_version : str | None
        Version of the primary ML framework (e.g. ``'2.1.0'``).
    torch_version : str | None
        PyTorch version if applicable.
    sklearn_version : str | None
        Scikit-learn version if applicable.
    dataset_checksum : str | None
        SHA-256 hex digest of the training dataset (or a deterministic
        sample for large datasets).
    dataset_name : str | None
        Human-readable dataset identifier.
    training_duration_seconds : float | None
        Wall-clock seconds spent training this model.
    output_schema : dict[str, Any] | None
        Description of the model's output shape and types, e.g.
        ``{"shape": [1, 10], "dtype": "float32"}``.
    model_type : str | None
        Model class name (e.g. ``'GraphSAGE'``, ``'DeepSVDD'``).
    input_schema : dict[str, Any] | None
        Description of the model's expected input shape and types.
    git_commit : str | None
        Git commit hash of the training run for full lineage.
    training_config : dict[str, Any] | None
        Hyperparameters / training config used.
    custom : dict[str, Any]
        Arbitrary user-defined metadata.
    registered_at : str
        ISO-8601 timestamp of when this metadata was created.
    """

    framework_version: str | None = None
    torch_version: str | None = None
    sklearn_version: str | None = None
    dataset_checksum: str | None = None
    dataset_name: str | None = None
    training_duration_seconds: float | None = None
    output_schema: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
    model_type: str | None = None
    git_commit: str | None = None
    training_config: dict[str, Any] | None = None
    custom: dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        """Deserialise from a dict, ignoring unknown keys for forward compat."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------


def detect_framework_versions() -> dict[str, str | None]:
    """Auto-detect installed ML framework versions.

    Returns
    -------
    dict with keys ``framework_version``, ``torch_version``,
    ``sklearn_version``.  Values are ``None`` if the framework is not
    installed.
    """
    torch_version = None
    sklearn_version = None
    framework_version = None

    try:
        import torch

        torch_version = torch.__version__
        framework_version = f"torch={torch_version}"
    except ImportError:
        pass

    try:
        import sklearn

        sklearn_version = sklearn.__version__
        if framework_version:
            framework_version += f",sklearn={sklearn_version}"
        else:
            framework_version = f"sklearn={sklearn_version}"
    except ImportError:
        pass

    return {
        "framework_version": framework_version,
        "torch_version": torch_version,
        "sklearn_version": sklearn_version,
    }


def compute_dataset_checksum(
    data: Any,
    max_samples: int = 10_000,
) -> str:
    """Compute a deterministic checksum for a training dataset.

    For large datasets only ``max_samples`` entries are used to keep
    the checksum fast.  The method is deterministic for the same
    sample.

    Parameters
    ----------
    data : list | numpy.ndarray | pandas.DataFrame
        The training data.
    max_samples : int
        Maximum number of elements to hash.

    Returns
    -------
    str : SHA-256 hex digest.
    """
    h = hashlib.sha256()

    if hasattr(data, "to_csv"):
        # pandas DataFrame
        sample = data.head(max_samples)
        h.update(sample.to_csv(index=False).encode("utf-8"))
    elif hasattr(data, "tobytes"):
        # numpy array
        arr = data[:max_samples] if len(data) > max_samples else data
        h.update(arr.tobytes())
    elif isinstance(data, (list, tuple)):
        for item in data[:max_samples]:
            h.update(str(item).encode("utf-8"))
    else:
        h.update(str(data).encode("utf-8"))

    return h.hexdigest()


def infer_output_schema(model: Any, sample_input: Any = None) -> dict[str, Any] | None:
    """Infer the output schema from a model and optional sample input.

    Parameters
    ----------
    model : torch.nn.Module or any model with ``__call__``
        The trained model.
    sample_input : optional
        A sample input tensor/array to run a forward pass with.

    Returns
    -------
    dict with ``shape`` and ``dtype`` keys, or ``None`` if inference
    fails.
    """
    try:
        import torch

        if isinstance(model, torch.nn.Module) and sample_input is not None:
            model.eval()
            with torch.no_grad():
                output = model(sample_input)
            return {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
            }
    except Exception:
        pass

    # Fallback: try to read from model attributes
    if hasattr(model, "output_dim"):
        return {"shape": [None, model.output_dim], "dtype": "float32"}
    if hasattr(model, "out_channels"):
        return {"shape": [None, model.out_channels], "dtype": "float32"}

    return None


def detect_git_commit() -> str | None:
    """Detect the current git commit hash, if available."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def collect_metadata(
    model: Any = None,
    dataset: Any = None,
    training_duration: float | None = None,
    training_config: dict[str, Any] | None = None,
    sample_input: Any = None,
    dataset_name: str | None = None,
    custom: dict[str, Any] | None = None,
) -> ArtifactMetadata:
    """Convenience function to collect all enriched metadata at once.

    Parameters
    ----------
    model : optional
        Trained model for output schema and type detection.
    dataset : optional
        Training dataset for checksum computation.
    training_duration : float | None
        Wall-clock seconds for training.
    training_config : dict | None
        Training hyperparameters.
    sample_input : optional
        Sample input for output schema inference.
    dataset_name : str | None
        Human-readable dataset name.
    custom : dict | None
        User-defined metadata.

    Returns
    -------
    ArtifactMetadata with all available fields populated.
    """
    versions = detect_framework_versions()

    dataset_checksum = None
    if dataset is not None:
        dataset_checksum = compute_dataset_checksum(dataset)

    output_schema = None
    model_type = None
    if model is not None:
        output_schema = infer_output_schema(model, sample_input)
        model_type = type(model).__name__

    return ArtifactMetadata(
        framework_version=versions["framework_version"],
        torch_version=versions["torch_version"],
        sklearn_version=versions["sklearn_version"],
        dataset_checksum=dataset_checksum,
        dataset_name=dataset_name,
        training_duration_seconds=training_duration,
        output_schema=output_schema,
        model_type=model_type,
        git_commit=detect_git_commit(),
        training_config=training_config,
        custom=custom or {},
    )


# ---------------------------------------------------------------------------
# Migration helpers for existing registry entries
# ---------------------------------------------------------------------------


def enrich_registry_entry(
    entry: dict[str, Any],
    model: Any = None,
    dataset: Any = None,
    training_duration: float | None = None,
) -> dict[str, Any]:
    """Enrich an existing registry entry dict with new metadata fields.

    Existing fields are preserved.  Only missing fields are added.

    Parameters
    ----------
    entry : dict
        Existing registry entry (as stored in ``meta.json``).
    model : optional
        Model to extract output schema / type from.
    dataset : optional
        Dataset for checksum.
    training_duration : float | None
        Training time in seconds.

    Returns
    -------
    Updated entry dict with enriched metadata.
    """
    metadata = collect_metadata(
        model=model,
        dataset=dataset,
        training_duration=training_duration,
    )

    # Merge — don't overwrite existing fields
    meta_dict = metadata.to_dict()
    for key, value in meta_dict.items():
        if key not in entry or entry[key] is None:
            entry[key] = value

    # Also merge into custom_metadata if present
    if "custom_metadata" in entry and isinstance(entry["custom_metadata"], dict):
        entry["custom_metadata"]["_enriched_at"] = metadata.registered_at
    elif "custom_metadata" not in entry:
        entry["custom_metadata"] = {"_enriched_at": metadata.registered_at}

    return entry


def migrate_registry_file(meta_path: str | Path) -> dict[str, Any] | None:
    """Read and enrich a ``*.meta.json`` file on disk.

    Parameters
    ----------
    meta_path : str | Path
        Path to the ``.meta.json`` sidecar file.

    Returns
    -------
    Enriched dict, or ``None`` if the file doesn't exist / can't be read.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read registry entry %s: %s", meta_path, e)
        return None

    enriched = enrich_registry_entry(entry)

    try:
        tmp_path = meta_path.with_suffix(".meta.json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2)
        os.replace(str(tmp_path), str(meta_path))
    except OSError as e:
        logger.warning("Failed to write enriched registry entry %s: %s", meta_path, e)

    return enriched
