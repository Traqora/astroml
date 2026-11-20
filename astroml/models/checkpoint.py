"""Versioned checkpoints for the baseline GCN (issue #1201).

A bare ``state_dict`` carries no record of the architecture that produced it, so
loading one into a model of a different width fails with a shape error that names
a tensor but not the cause, and loading one saved by a newer release can quietly
do the wrong thing. This module defines the file format and says exactly when a
checkpoint is compatible.

Format (version 1)
------------------
A ``torch.save`` file holding a dict with only tensors, numbers and strings, so it
loads with ``torch.load(..., weights_only=True)`` and never executes pickled code:

``format``
    The constant :data:`CHECKPOINT_FORMAT`.
``version``
    Integer format version, :data:`CHECKPOINT_VERSION` when written.
``config``
    ``input_dim``, ``hidden_dim``, ``output_dim`` and ``dropout``.
``state_dict``
    The model weights.
``metadata``
    JSON-safe values: the environment the checkpoint was written in (see
    :mod:`astroml.utils.environment`) and anything the caller passes.

Compatibility rules
-------------------
:func:`load_gcn_checkpoint` accepts a checkpoint only if all of these hold, and
otherwise raises :class:`CheckpointCompatibilityError` naming the one that failed:

1. The file is an AstroML GCN checkpoint (``format`` matches).
2. Its ``version`` is between :data:`MIN_SUPPORTED_VERSION` and
   :data:`CHECKPOINT_VERSION`. A newer version is refused rather than guessed at.
3. Its ``config`` is complete and valid.
4. If the caller passes ``expected``, the architecture matches it
   (``input_dim``, ``hidden_dim`` and ``output_dim``). ``dropout`` is not part of
   the architecture: it holds no weights, so a difference is allowed.
5. Its ``state_dict`` has exactly the keys and shapes of a model built from
   ``config``.

Writes are atomic: the file is written beside the target and renamed into place, so
an interruption never leaves a truncated checkpoint where a good one used to be.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import torch

from astroml.models.gcn import GCN
from astroml.utils.environment import collect_environment
from astroml.utils.exceptions import ModelError

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "MIN_SUPPORTED_VERSION",
    "CheckpointCompatibilityError",
    "GCNConfig",
    "LoadedCheckpoint",
    "gcn_config",
    "load_gcn_checkpoint",
    "save_gcn_checkpoint",
]

CHECKPOINT_FORMAT = "astroml.gcn.checkpoint"
CHECKPOINT_VERSION = 1
MIN_SUPPORTED_VERSION = 1


class CheckpointCompatibilityError(ModelError):
    """A checkpoint cannot be loaded into the requested model.

    The internal message says which compatibility rule failed; clients only see the
    stable code and the generic public message.
    """

    code: ClassVar[str] = "CHECKPOINT_INCOMPATIBLE"
    public_message: ClassVar[str] = "The model checkpoint is not compatible with this version."


@dataclass(frozen=True)
class GCNConfig:
    """Constructor arguments that fix a :class:`~astroml.models.gcn.GCN`'s weights.

    Attributes:
        input_dim: Node feature width.
        hidden_dim: Hidden layer width.
        output_dim: Number of classes.
        dropout: Dropout probability. Not part of the architecture.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float = 0.5

    def architecture(self) -> tuple[int, int, int]:
        """The dimensions that decide weight shapes, excluding ``dropout``."""
        return (self.input_dim, self.hidden_dim, self.output_dim)

    def to_dict(self) -> dict[str, Any]:
        """Plain-value form stored in the checkpoint."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }

    def build(self) -> GCN:
        """A freshly initialised model with this configuration."""
        return GCN(self.input_dim, self.hidden_dim, self.output_dim, dropout=self.dropout)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """The result of a successful :func:`load_gcn_checkpoint`.

    Attributes:
        model: The model, weights loaded, in evaluation mode.
        config: The configuration read from the file.
        version: The format version the file was written with.
        metadata: The metadata stored with the checkpoint.
    """

    model: GCN
    config: GCNConfig
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)


def gcn_config(model: GCN) -> GCNConfig:
    """Read a model's configuration back from its layers."""
    return GCNConfig(
        input_dim=int(model.conv1.in_channels),
        hidden_dim=int(model.conv1.out_channels),
        output_dim=int(model.conv2.out_channels),
        dropout=float(model.dropout),
    )


def save_gcn_checkpoint(
    model: GCN, path: str | os.PathLike[str], *, metadata: Mapping[str, Any] | None = None
) -> Path:
    """Write ``model`` and its configuration to ``path`` atomically.

    Args:
        model: The model to save.
        path: Destination file.
        metadata: Extra JSON-safe values to store. The current environment is
            always recorded under ``environment``.

    Returns:
        The path written.

    Raises:
        ValueError: If ``metadata`` cannot be represented as JSON. Checking here
            keeps the file loadable with ``weights_only=True``.
    """
    extra = dict(metadata or {})
    try:
        json.dumps(extra)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint metadata must be JSON-serialisable: {error}") from error

    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": gcn_config(model).to_dict(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {"environment": collect_environment(), **extra},
    }

    target = Path(path)
    staging = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, staging)
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()
    return target


def _fail(path: Path, reason: str) -> CheckpointCompatibilityError:
    return CheckpointCompatibilityError(f"{path}: {reason}", path=str(path))


def _parse_config(path: Path, raw: Any) -> GCNConfig:
    if not isinstance(raw, Mapping):
        raise _fail(path, "checkpoint has no 'config' mapping")

    dims: dict[str, int] = {}
    for key in ("input_dim", "hidden_dim", "output_dim"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _fail(path, f"config.{key} must be a positive integer, got {value!r}")
        dims[key] = value

    dropout = raw.get("dropout", 0.5)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise _fail(path, f"config.dropout must be in [0, 1), got {dropout!r}")

    return GCNConfig(dropout=float(dropout), **dims)


def load_gcn_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected: GCNConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a GCN checkpoint, enforcing the compatibility rules in the module docs.

    Args:
        path: Checkpoint file.
        expected: The architecture the caller needs. When given, a checkpoint with
            different ``input_dim``, ``hidden_dim`` or ``output_dim`` is refused.
        map_location: Device to load tensors onto.

    Returns:
        The loaded model (in evaluation mode) with its configuration and metadata.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        CheckpointCompatibilityError: If the file is unreadable or breaks a
            compatibility rule.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as error:  # noqa: BLE001 - torch raises several unrelated types here
        raise _fail(source, f"file could not be read as a checkpoint ({error})") from error

    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise _fail(source, f"not an {CHECKPOINT_FORMAT} file")

    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise _fail(source, f"checkpoint version must be an integer, got {version!r}")
    if version > CHECKPOINT_VERSION:
        raise _fail(
            source,
            f"written with format version {version}, newer than the {CHECKPOINT_VERSION} "
            "this release reads; upgrade AstroML to load it",
        )
    if version < MIN_SUPPORTED_VERSION:
        raise _fail(
            source,
            f"format version {version} is older than the minimum supported "
            f"version {MIN_SUPPORTED_VERSION}",
        )

    config = _parse_config(source, payload.get("config"))

    if expected is not None and config.architecture() != expected.architecture():
        raise _fail(
            source,
            "architecture mismatch: checkpoint has "
            f"(input_dim, hidden_dim, output_dim)={config.architecture()}, "
            f"expected {expected.architecture()}",
        )

    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise _fail(source, "checkpoint has no 'state_dict' mapping")

    model = config.build()
    reference = model.state_dict()
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    wrong_shape = sorted(
        key
        for key in set(reference) & set(state)
        if tuple(state[key].shape) != tuple(reference[key].shape)
    )
    if missing or unexpected or wrong_shape:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unexpected:
            details.append(f"unexpected keys {unexpected}")
        if wrong_shape:
            details.append(f"wrong shape for {wrong_shape}")
        raise _fail(source, "state_dict does not match the configuration: " + "; ".join(details))

    model.load_state_dict(state)
    model.eval()

    raw_metadata = payload.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    return LoadedCheckpoint(model=model, config=config, version=version, metadata=metadata)
