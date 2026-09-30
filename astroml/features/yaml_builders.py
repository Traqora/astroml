"""YAML-configurable feature builders.

Issue #742: makes feature builders declaratively configurable so new features
can be added without writing code. A builder definition lives in a YAML file
(see ``configs/feature_builders.yaml``) describing the name, inputs, window,
output column, and the computation primitive to apply. Definitions are loaded
into :class:`~astroml.features.feature_registry.FeatureDefinition` objects and
wired to a primitive registry at runtime.

Example YAML::

    builders:
      - name: tx_count_7d
        description: Number of transactions in a trailing 7-day window
        primitive: count
        feature_type: numeric
        inputs:
          entity: account
          source: transactions
          columns: [timestamp, amount]
        window: 7d
        output_column: tx_count_7d
        tags: [temporal, usage]

Caching / recompute semantics (also relied on by issue #743):
    - Every builder records the window and any parameters that affect output,
      so identical definitions always produce identical cache keys.
    - Changing the window or parameters in YAML changes the definition hash,
      which invalidates previously cached values on the next build (the cache
      entry is keyed with the new hash and the old entry simply goes stale).
    - Recompute happens when: the definition hash changes, the data version
      changes, or the cache entry is expired/evicted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from astroml.features.feature_registry import (
    FeatureDefinition,
    FeatureRegistryService,
    FeatureStatus,
    FeatureType,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BuilderSpecError",
    "FEATURE_PRIMITIVES",
    "FeatureBuilderSpec",
    "load_feature_builder_specs",
    "load_feature_builders",
    "register_feature_builder_specs",
]


class BuilderSpecError(ValueError):
    """Raised when a YAML builder definition is invalid."""


# ---------------------------------------------------------------------------
# Computation primitives
# ---------------------------------------------------------------------------


def _require_columns(columns: Sequence[str], row: Mapping[str, Any]) -> None:
    """Raise BuilderSpecError if any required column is missing from a row."""
    for column in columns:
        if column not in row:
            raise BuilderSpecError(f"input column {column!r} missing from data")


def _primitive_count(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Count of input rows inside the window."""
    return float(len(rows))


def _primitive_sum(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Sum of ``value_column`` (default ``amount``) over the window."""
    column = spec.params.get("value_column", "amount")
    total = 0.0
    for row in rows:
        _require_columns([column], row)
        total += float(row[column])
    return total


def _primitive_mean(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Arithmetic mean of ``value_column`` (default ``amount``) over the window."""
    if not rows:
        return 0.0
    return _primitive_sum(rows, spec) / len(rows)


def _primitive_max(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Maximum of ``value_column`` (default ``amount``) over the window."""
    if not rows:
        return 0.0
    column = spec.params.get("value_column", "amount")
    return max(float(row[column]) for row in rows)


def _primitive_min(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Minimum of ``value_column`` (default ``amount``) over the window."""
    if not rows:
        return 0.0
    column = spec.params.get("value_column", "amount")
    return min(float(row[column]) for row in rows)


def _primitive_std(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Population standard deviation of ``value_column`` over the window."""
    if len(rows) < 2:
        return 0.0
    column = spec.params.get("value_column", "amount")
    values = [float(row[column]) for row in rows]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5


def _primitive_unique_count(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Number of distinct values of ``value_column`` over the window."""
    column = spec.params.get("value_column", "counterparty")
    return float(len({row.get(column) for row in rows}))


def _primitive_ratio(rows: list[dict[str, Any]], spec: FeatureBuilderSpec) -> float:
    """Numerator column sum divided by denominator column sum (0 when empty)."""
    numerator = spec.params.get("numerator", "amount")
    denominator = spec.params.get("denominator", "amount")
    top = 0.0
    bottom = 0.0
    for row in rows:
        _require_columns([numerator, denominator], row)
        top += float(row[numerator])
        bottom += float(row[denominator])
    if bottom == 0:
        return 0.0
    return top / bottom


#: Built-in primitives available to YAML definitions. Custom primitives can be
#: added by passing ``primitives={...}`` to ``register_feature_builder_specs``.
FEATURE_PRIMITIVES: dict[str, Callable[[list[dict[str, Any]], FeatureBuilderSpec], float]] = {
    "count": _primitive_count,
    "sum": _primitive_sum,
    "mean": _primitive_mean,
    "max": _primitive_max,
    "min": _primitive_min,
    "std": _primitive_std,
    "unique_count": _primitive_unique_count,
    "ratio": _primitive_ratio,
}


# ---------------------------------------------------------------------------
# Builder specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBuilderSpec:
    """A single declarative feature-builder definition parsed from YAML.

    Attributes:
        name: Unique feature name.
        description: Human-readable description (used for docs and registry).
        primitive: Name of a computation primitive in a primitives mapping.
        inputs: Mapping describing inputs, e.g. ``entity``, ``source``,
            ``columns`` (the columns the computation reads).
        window: Trailing window string such as ``7d`` (days), ``12h`` (hours),
            or ``30m`` (minutes). Omit for windowless features.
        output_column: Column name the built feature is written to.
        feature_type: Registry feature type (numeric by default).
        tags: Arbitrary tags carried into the registry definition.
        owner: Owner recorded in the registry definition.
        params: Extra primitive-specific parameters.
    """

    name: str
    description: str
    primitive: str
    inputs: dict[str, Any] = field(default_factory=dict)
    window: str | None = None
    output_column: str | None = None
    feature_type: FeatureType = FeatureType.NUMERIC
    tags: list[str] = field(default_factory=list)
    owner: str = "system"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise BuilderSpecError("builder 'name' must be a non-empty string")
        if not self.primitive:
            raise BuilderSpecError(f"builder {self.name!r} is missing 'primitive'")
        if self.window is not None:
            _parse_window(self.window)

    @property
    def window_seconds(self) -> float | None:
        """Window length in seconds, or None when windowless."""
        if self.window is None:
            return None
        return _parse_window(self.window)

    @property
    def columns(self) -> list[str]:
        """Input columns this builder reads (from ``inputs.columns``)."""
        columns = self.inputs.get("columns", [])
        return [str(c) for c in columns]

    @property
    def source(self) -> str | None:
        """Logical input source (e.g. ``transactions``), if declared."""
        source = self.inputs.get("source")
        return str(source) if source is not None else None

    @property
    def entity(self) -> str | None:
        """Logical entity (e.g. ``account``), if declared."""
        entity = self.inputs.get("entity")
        return str(entity) if entity is not None else None

    def definition_hash(self) -> str:
        """Stable hash of everything that affects this builder's output.

        Used for cache keys (#743): changing the window, primitive, inputs, or
        parameters produces a different hash, which invalidates stale entries.
        """
        payload = {
            "name": self.name,
            "primitive": self.primitive,
            "window": self.window,
            "inputs": self.inputs,
            "params": self.params,
            "output_column": self.output_column,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def to_registry_definition(self) -> FeatureDefinition:
        """Convert to a registry :class:`FeatureDefinition`."""
        return FeatureDefinition(
            name=self.name,
            description=self.description,
            feature_type=self.feature_type,
            parameters={
                "primitive": self.primitive,
                "window": self.window,
                "inputs": self.inputs,
                "output_column": self.output_column,
                **self.params,
            },
            tags=list(self.tags) + ["yaml_builder"],
            owner=self.owner,
            status=FeatureStatus.PRODUCTION,
            metadata={"definition_hash": self.definition_hash()},
        )


def _parse_window(window: str) -> float:
    """Parse a window string (``7d``, ``12h``, ``30m``) into seconds."""
    text = str(window).strip().lower()
    units = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    if len(text) < 2 or text[-1] not in units or not text[:-1].replace(".", "", 1).isdigit():
        raise BuilderSpecError(
            f"invalid window {window!r}; expected a number followed by d, h, m, or s"
        )
    return float(text[:-1]) * units[text[-1]]


# ---------------------------------------------------------------------------
# YAML loading and registration
# ---------------------------------------------------------------------------


def load_feature_builder_specs(path: str | Path) -> list[FeatureBuilderSpec]:
    """Load builder specs from a YAML file.

    The file must contain a top-level ``builders`` list. Unknown keys are
    rejected so typos fail loudly instead of silently changing semantics.

    Raises:
        BuilderSpecError: If the file structure or any definition is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise BuilderSpecError(f"builder definition file not found: {file_path}")

    try:
        raw = yaml.safe_load(file_path.read_text())
    except yaml.YAMLError as exc:
        raise BuilderSpecError(f"invalid YAML in {file_path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("builders"), list):
        raise BuilderSpecError(f"{file_path} must contain a top-level 'builders' list")

    specs: list[FeatureBuilderSpec] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(raw["builders"]):
        if not isinstance(entry, dict):
            raise BuilderSpecError(f"builders[{index}] must be a mapping")
        try:
            spec = _spec_from_dict(entry)
        except BuilderSpecError as exc:
            raise BuilderSpecError(f"builders[{index}]: {exc}") from exc
        if spec.name in seen_names:
            raise BuilderSpecError(f"duplicate builder name: {spec.name!r}")
        seen_names.add(spec.name)
        specs.append(spec)

    logger.info("Loaded %d feature builder specs from %s", len(specs), file_path)
    return specs


def _spec_from_dict(entry: dict[str, Any]) -> FeatureBuilderSpec:
    """Build a FeatureBuilderSpec from a validated mapping."""
    allowed = {
        "name",
        "description",
        "primitive",
        "inputs",
        "window",
        "output_column",
        "feature_type",
        "tags",
        "owner",
        "params",
    }
    unknown = set(entry) - allowed
    if unknown:
        raise BuilderSpecError(f"unknown keys: {sorted(unknown)}")
    if "primitive" not in entry:
        raise BuilderSpecError("missing required key 'primitive'")

    feature_type = entry.get("feature_type", "numeric")
    if isinstance(feature_type, str):
        try:
            feature_type = FeatureType(feature_type)
        except ValueError as exc:
            raise BuilderSpecError(f"invalid feature_type {feature_type!r}") from exc

    return FeatureBuilderSpec(
        name=str(entry.get("name", "")),
        description=str(entry.get("description", "")),
        primitive=str(entry["primitive"]),
        inputs=dict(entry.get("inputs") or {}),
        window=entry.get("window"),
        output_column=entry.get("output_column"),
        feature_type=feature_type,
        tags=[str(t) for t in entry.get("tags", [])],
        owner=str(entry.get("owner", "system")),
        params=dict(entry.get("params") or {}),
    )


def load_feature_builders(
    path: str | Path,
    primitives: (
        Mapping[str, Callable[[list[dict[str, Any]], FeatureBuilderSpec], float]] | None
    ) = None,
) -> list[tuple[FeatureBuilderSpec, Callable[[list[dict[str, Any]], FeatureBuilderSpec], float]]]:
    """Load builder specs from YAML and resolve each to its primitive.

    Args:
        path: Path to the YAML definitions file.
        primitives: Optional custom primitives mapping; when omitted the
            built-in :data:`FEATURE_PRIMITIVES` are used.

    Returns:
        List of ``(spec, primitive_callable)`` pairs, in file order.

    Raises:
        BuilderSpecError: If a definition references an unknown primitive.
    """
    # An explicit mapping replaces the built-ins entirely so callers can
    # restrict the available primitives (pass {} to allow none).
    known = dict(primitives) if primitives is not None else dict(FEATURE_PRIMITIVES)

    pairs: list[tuple[FeatureBuilderSpec, Callable[..., float]]] = []
    for spec in load_feature_builder_specs(path):
        primitive = known.get(spec.primitive)
        if primitive is None:
            raise BuilderSpecError(
                f"builder {spec.name!r} references unknown primitive {spec.primitive!r}; "
                f"known primitives: {sorted(known)}"
            )
        pairs.append((spec, primitive))
    return pairs


def register_feature_builder_specs(
    registry: FeatureRegistryService,
    specs: Sequence[FeatureBuilderSpec],
    primitives: (
        Mapping[str, Callable[[list[dict[str, Any]], FeatureBuilderSpec], float]] | None
    ) = None,
) -> list[FeatureDefinition]:
    """Register builder specs (and their computers) with a feature registry.

    Args:
        registry: The registry service to register into.
        specs: Specs previously loaded via ``load_feature_builder_specs``.
        primitives: Optional custom primitives mapping used to resolve and
            attach computation callables.

    Returns:
        The created FeatureDefinition objects, in registration order.
    """
    # See load_feature_builders: an explicit mapping replaces the built-ins.
    known = dict(primitives) if primitives is not None else dict(FEATURE_PRIMITIVES)

    definitions: list[FeatureDefinition] = []
    for spec in specs:
        primitive = known.get(spec.primitive)
        if primitive is None:
            raise BuilderSpecError(
                f"builder {spec.name!r} references unknown primitive {spec.primitive!r}"
            )

        def _compute(
            rows: list[dict[str, Any]],
            builder_spec: FeatureBuilderSpec = spec,
            fn: Callable[[list[dict[str, Any]], FeatureBuilderSpec], float] = primitive,
            **_kwargs: Any,
        ) -> float:
            return fn(rows, builder_spec)

        definitions.append(
            registry.register(
                name=spec.name,
                computer=_compute,
                description=spec.description,
                feature_type=spec.feature_type,
                tags=list(spec.tags) + ["yaml_builder"],
                owner=spec.owner,
                parameters={
                    "primitive": spec.primitive,
                    "window": spec.window,
                    "inputs": spec.inputs,
                    "output_column": spec.output_column,
                    **spec.params,
                },
                metadata={"definition_hash": spec.definition_hash(), "source": "yaml"},
            )
        )
    return definitions
