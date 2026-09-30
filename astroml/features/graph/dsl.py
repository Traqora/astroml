"""Declarative graph-builder spec — issue #739.

A snapshot is currently described by whatever arguments the calling script
happened to pass: window size here, an amount filter there, a hop count
somewhere else. Reproducing last month's experiment then means reading last
month's script. This module makes the description a document instead:

.. code-block:: yaml

    version: 1
    name: weekly-payments
    window:
      size: 7d
      step: 1d
    edges:
      types: [payment, path_payment]
      min_amount: 10.0
      exclude_self_loops: true
      directed: true
    features:
      max_hops: 2
      neighbour_aggregates: true
    nodes:
      attributes: [degree, total_sent, hop2_size]

:meth:`GraphSpec.from_yaml` parses and validates it, :meth:`GraphSpec.fingerprint`
gives a stable identifier to record with the run, and :func:`build_from_spec`
turns spec plus edges into a snapshot, its node features and its statistics.

Validation is strict and total: unknown keys are rejected rather than
ignored (a spec that silently drops ``mim_amount`` is worse than one that
refuses to load), and every problem in the document is reported at once
instead of one exception per run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .neighborhood import (
    NeighborhoodConfig,
    NeighborhoodFeatures,
    _base_feature_names,
    normalize_edges,
    precompute_neighborhood_features,
)
from .statistics import SnapshotStats, compute_snapshot_stats

#: The only spec version this module understands.
SPEC_VERSION = 1

_DURATION = re.compile(r"^(\d+)([smhd])$")
_DURATION_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

_TOP_LEVEL_KEYS = {"version", "name", "window", "edges", "features", "nodes"}
_WINDOW_KEYS = {"size", "step", "t0", "t_now"}
_EDGE_KEYS = {"types", "min_amount", "max_amount", "exclude_self_loops", "directed"}
_FEATURE_KEYS = {"max_hops", "neighbour_aggregates", "hop_node_budget"}
_NODE_KEYS = {"attributes"}


class SpecValidationError(ValueError):
    """Raised when a graph spec is not usable.

    Carries every problem found, not just the first, so a contributor
    editing a config fixes the whole document in one pass.
    """

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def parse_duration(value: str) -> int:
    """Parse ``'7d'``/``'24h'``/``'30m'``/``'3600s'`` into seconds."""
    match = _DURATION.match(str(value).strip())
    if not match:
        raise ValueError(f"invalid duration {value!r}; expected a number followed by s, m, h or d")
    amount, unit = match.groups()
    seconds = int(amount) * _DURATION_SECONDS[unit]
    if seconds <= 0:
        raise ValueError(f"duration must be positive, got {value!r}")
    return seconds


@dataclass(frozen=True)
class WindowSpec:
    """How the timeline is sliced."""

    size: str = "7d"
    step: str | None = None
    t0: str | None = None
    t_now: str | None = None

    @property
    def size_seconds(self) -> int:
        return parse_duration(self.size)

    @property
    def step_seconds(self) -> int:
        return parse_duration(self.step) if self.step else self.size_seconds

    @property
    def overlapping(self) -> bool:
        return self.step_seconds < self.size_seconds


@dataclass(frozen=True)
class EdgeSpec:
    """Which edges are admitted into the graph."""

    types: tuple[str, ...] = ()
    min_amount: float | None = None
    max_amount: float | None = None
    exclude_self_loops: bool = True
    directed: bool = True

    def accepts(self, edge: Mapping[str, Any]) -> bool:
        """Whether ``edge`` passes this filter."""
        if self.types:
            edge_type = edge.get("type", edge.get("edge_type"))
            if edge_type is None or str(edge_type) not in self.types:
                return False

        src = edge.get("src", edge.get("source"))
        dst = edge.get("dst", edge.get("destination"))
        if self.exclude_self_loops and src == dst:
            return False

        amount = edge.get("amount")
        if amount is not None:
            amount = float(amount)
            if self.min_amount is not None and amount < self.min_amount:
                return False
            if self.max_amount is not None and amount > self.max_amount:
                return False
        elif self.min_amount is not None:
            # An amount filter cannot be satisfied by an edge that carries no
            # amount; admitting it would quietly widen the spec.
            return False

        return True


@dataclass(frozen=True)
class FeatureSpec:
    """Which neighbourhood features to precompute."""

    max_hops: int = 2
    neighbour_aggregates: bool = True
    hop_node_budget: int | None = 10_000


@dataclass(frozen=True)
class NodeSpec:
    """Which of the computed features end up on the nodes.

    Empty means "all of them", which is the common case; naming a subset
    pins the feature matrix so a later feature addition cannot silently
    change a model's input width.
    """

    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphSpec:
    """A complete, validated graph-building recipe."""

    name: str
    version: int = SPEC_VERSION
    window: WindowSpec = field(default_factory=WindowSpec)
    edges: EdgeSpec = field(default_factory=EdgeSpec)
    features: FeatureSpec = field(default_factory=FeatureSpec)
    nodes: NodeSpec = field(default_factory=NodeSpec)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GraphSpec:
        """Build a spec from an already-parsed mapping.

        Raises:
            SpecValidationError: with every problem found in the document.
        """
        errors: list[str] = []

        if not isinstance(payload, Mapping):
            raise SpecValidationError(["spec must be a mapping at the top level"])

        _reject_unknown(payload, _TOP_LEVEL_KEYS, "", errors)

        version = payload.get("version", SPEC_VERSION)
        if not isinstance(version, int) or isinstance(version, bool):
            errors.append(f"version must be an integer, got {version!r}")
        elif version != SPEC_VERSION:
            errors.append(
                f"unsupported spec version {version}; this build understands {SPEC_VERSION}"
            )

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append("name is required and must be a non-empty string")
            name = ""

        window = _window_from(payload.get("window", {}), errors)
        edges = _edges_from(payload.get("edges", {}), errors)
        features = _features_from(payload.get("features", {}), errors)
        nodes = _nodes_from(payload.get("nodes", {}), errors)

        # Cross-field: a named attribute must actually be produced by the
        # feature settings in this same document.
        if nodes.attributes:
            available = set(
                _base_feature_names(
                    NeighborhoodConfig(
                        max_hops=max(features.max_hops, 1),
                        neighbour_aggregates=features.neighbour_aggregates,
                    )
                )
            )
            for attribute in nodes.attributes:
                if attribute not in available:
                    errors.append(
                        f"nodes.attributes: {attribute!r} is not produced by this "
                        f"feature config (available: {', '.join(sorted(available))})"
                    )

        if errors:
            raise SpecValidationError(errors)

        return cls(
            name=name.strip(),
            version=SPEC_VERSION,
            window=window,
            edges=edges,
            features=features,
            nodes=nodes,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraphSpec:
        """Load and validate a spec from a YAML file."""
        import yaml

        source = Path(path)
        if not source.is_file():
            raise SpecValidationError([f"spec file not found: {source}"])
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SpecValidationError([f"{source} is not valid YAML: {exc}"]) from exc
        if payload is None:
            raise SpecValidationError([f"{source} is empty"])
        return cls.from_mapping(payload)

    # -- identity --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "window": {
                "size": self.window.size,
                "step": self.window.step,
                "t0": self.window.t0,
                "t_now": self.window.t_now,
            },
            "edges": {
                "types": list(self.edges.types),
                "min_amount": self.edges.min_amount,
                "max_amount": self.edges.max_amount,
                "exclude_self_loops": self.edges.exclude_self_loops,
                "directed": self.edges.directed,
            },
            "features": {
                "max_hops": self.features.max_hops,
                "neighbour_aggregates": self.features.neighbour_aggregates,
                "hop_node_budget": self.features.hop_node_budget,
            },
            "nodes": {"attributes": list(self.nodes.attributes)},
        }

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=True)

    def fingerprint(self) -> str:
        """Stable identifier for this spec.

        Record it with the experiment: two runs with the same fingerprint
        built their graphs the same way, whatever the surrounding script
        looked like. The spec ``name`` is included, so renaming an
        experiment is a visible change rather than an invisible one.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def neighborhood_config(self) -> NeighborhoodConfig:
        return NeighborhoodConfig(
            max_hops=self.features.max_hops,
            directed=self.edges.directed,
            neighbour_aggregates=self.features.neighbour_aggregates,
            hop_node_budget=self.features.hop_node_budget,
        )


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    prefix: str,
    errors: list[str],
) -> None:
    for key in payload:
        if key not in allowed:
            where = f"{prefix}{key}" if prefix else str(key)
            errors.append(
                f"unknown key {where!r} (did you mean one of: {', '.join(sorted(allowed))}?)"
            )


def _as_section(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping, got {type(value).__name__}")
        return {}
    return value


def _window_from(value: Any, errors: list[str]) -> WindowSpec:
    section = _as_section(value, "window", errors)
    _reject_unknown(section, _WINDOW_KEYS, "window.", errors)

    size = section.get("size", "7d")
    step = section.get("step")
    for label, duration in (("window.size", size), ("window.step", step)):
        if duration is None:
            continue
        try:
            parse_duration(duration)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")

    try:
        if step is not None and parse_duration(step) > parse_duration(size):
            errors.append("window.step must not exceed window.size — that would skip edges")
    except ValueError:
        pass  # already reported above

    return WindowSpec(
        size=str(size),
        step=None if step is None else str(step),
        t0=None if section.get("t0") is None else str(section["t0"]),
        t_now=None if section.get("t_now") is None else str(section["t_now"]),
    )


def _edges_from(value: Any, errors: list[str]) -> EdgeSpec:
    section = _as_section(value, "edges", errors)
    _reject_unknown(section, _EDGE_KEYS, "edges.", errors)

    raw_types = section.get("types", [])
    types: tuple[str, ...] = ()
    if raw_types is None:
        types = ()
    elif isinstance(raw_types, str):
        errors.append("edges.types must be a list, not a single string")
    elif isinstance(raw_types, Iterable):
        collected = [str(item) for item in raw_types]
        if len(set(collected)) != len(collected):
            errors.append(f"edges.types contains duplicates: {collected}")
        types = tuple(collected)
    else:
        errors.append(f"edges.types must be a list, got {type(raw_types).__name__}")

    bounds: dict[str, float | None] = {}
    for key in ("min_amount", "max_amount"):
        raw = section.get(key)
        if raw is None:
            bounds[key] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append(f"edges.{key} must be a number, got {raw!r}")
            bounds[key] = None
            continue
        if raw < 0:
            errors.append(f"edges.{key} must not be negative, got {raw}")
        bounds[key] = float(raw)

    if (
        bounds.get("min_amount") is not None
        and bounds.get("max_amount") is not None
        and bounds["min_amount"] > bounds["max_amount"]
    ):
        errors.append(
            f"edges.min_amount ({bounds['min_amount']}) exceeds "
            f"edges.max_amount ({bounds['max_amount']}) — no edge can match"
        )

    return EdgeSpec(
        types=types,
        min_amount=bounds.get("min_amount"),
        max_amount=bounds.get("max_amount"),
        exclude_self_loops=_as_bool(
            section, "exclude_self_loops", True, "edges.exclude_self_loops", errors
        ),
        directed=_as_bool(section, "directed", True, "edges.directed", errors),
    )


def _features_from(value: Any, errors: list[str]) -> FeatureSpec:
    section = _as_section(value, "features", errors)
    _reject_unknown(section, _FEATURE_KEYS, "features.", errors)

    max_hops = section.get("max_hops", 2)
    if isinstance(max_hops, bool) or not isinstance(max_hops, int):
        errors.append(f"features.max_hops must be an integer, got {max_hops!r}")
        max_hops = 2
    elif max_hops < 1:
        errors.append(f"features.max_hops must be >= 1, got {max_hops}")
        max_hops = 1

    budget = section.get("hop_node_budget", 10_000)
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int):
            errors.append(f"features.hop_node_budget must be an integer or null, got {budget!r}")
            budget = 10_000
        elif budget < 1:
            errors.append(f"features.hop_node_budget must be >= 1, got {budget}")
            budget = 1

    return FeatureSpec(
        max_hops=max_hops,
        neighbour_aggregates=_as_bool(
            section, "neighbour_aggregates", True, "features.neighbour_aggregates", errors
        ),
        hop_node_budget=budget,
    )


def _nodes_from(value: Any, errors: list[str]) -> NodeSpec:
    section = _as_section(value, "nodes", errors)
    _reject_unknown(section, _NODE_KEYS, "nodes.", errors)

    raw = section.get("attributes", [])
    if raw is None:
        return NodeSpec()
    if isinstance(raw, str):
        errors.append("nodes.attributes must be a list, not a single string")
        return NodeSpec()
    if not isinstance(raw, Iterable):
        errors.append(f"nodes.attributes must be a list, got {type(raw).__name__}")
        return NodeSpec()

    collected = [str(item) for item in raw]
    if len(set(collected)) != len(collected):
        errors.append(f"nodes.attributes contains duplicates: {collected}")
    return NodeSpec(attributes=tuple(collected))


def _as_bool(
    section: Mapping[str, Any],
    key: str,
    default: bool,
    label: str,
    errors: list[str],
) -> bool:
    raw = section.get(key, default)
    if not isinstance(raw, bool):
        errors.append(f"{label} must be a boolean, got {raw!r}")
        return default
    return raw


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltGraph:
    """The result of applying a spec to a set of edges."""

    spec_name: str
    spec_fingerprint: str
    edges: tuple[tuple[str, str, float], ...]
    nodes: tuple[str, ...]
    features: NeighborhoodFeatures
    stats: SnapshotStats
    num_rejected: int

    def feature_matrix(self) -> tuple[tuple[str, ...], list[list[float]], tuple[str, ...]]:
        """``(node_ids, rows, column_names)`` restricted to the spec's attributes."""
        columns = self.features.feature_names
        rows = [
            [self.features.values[node][name] for name in columns] for node in self.features.nodes
        ]
        return self.features.nodes, rows, columns


def build_from_spec(
    spec: GraphSpec,
    edges: Iterable[Mapping[str, Any]],
    index: int | None = None,
) -> BuiltGraph:
    """Apply ``spec`` to ``edges`` and return the built graph.

    Deterministic by construction: edges are filtered in the order given,
    then canonically sorted before any statistic or feature is computed, so
    the result depends on the spec and the edge *set* — never on the order
    the rows arrived in.
    """
    accepted: list[Mapping[str, Any]] = []
    rejected = 0
    for edge in edges:
        if spec.edges.accepts(edge):
            accepted.append(edge)
        else:
            rejected += 1

    normalized = normalize_edges(accepted)
    config = spec.neighborhood_config()
    features = precompute_neighborhood_features(normalized_to_mappings(normalized), config)

    if spec.nodes.attributes:
        keep = tuple(name for name in features.feature_names if name in spec.nodes.attributes)
        features = NeighborhoodFeatures(
            version=features.version,
            config_fingerprint=features.config_fingerprint,
            graph_fingerprint=features.graph_fingerprint,
            nodes=features.nodes,
            feature_names=keep,
            values={
                node: {name: row[name] for name in keep} for node, row in features.values.items()
            },
        )

    stats = compute_snapshot_stats(
        accepted,
        index=index,
        window_start=spec.window.t0,
        window_end=spec.window.t_now,
        nodes=features.nodes,
    )

    return BuiltGraph(
        spec_name=spec.name,
        spec_fingerprint=spec.fingerprint(),
        edges=tuple(normalized),
        nodes=features.nodes,
        features=features,
        stats=stats,
        num_rejected=rejected,
    )


def normalized_to_mappings(
    normalized: Sequence[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    """Re-present canonicalised triples as the mappings the feature layer takes."""
    return [{"src": src, "dst": dst, "amount": amount} for src, dst, amount in normalized]


__all__ = [
    "SPEC_VERSION",
    "BuiltGraph",
    "EdgeSpec",
    "FeatureSpec",
    "GraphSpec",
    "NodeSpec",
    "SpecValidationError",
    "WindowSpec",
    "build_from_spec",
    "parse_duration",
]
