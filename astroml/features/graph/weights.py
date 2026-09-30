"""Weighted edges for the graph snapshot builder — issue #731.

The snapshot builder yields one :class:`~astroml.features.graph.snapshot.Edge`
per transaction, so two accounts that traded a hundred times appear as a
hundred identical edges. Any model that wants "how strongly are these two
connected" has to collapse those itself, and every caller collapses them
slightly differently.

This module makes the collapse explicit. A :class:`WeightSpec` says where the
weight comes from (the transferred amount, how recent the activity is, or
just how many times it happened) and how weights for a repeated edge combine
(sum, mean, max, min, or most-recent). :func:`build_weighted_snapshot`
applies it and returns one weighted edge per node pair.

Determinism is the reason the combining policy is a parameter rather than a
convention: ``sum`` and ``last`` disagree on the same input, so a snapshot is
only reproducible if the policy travels with it. Edges are canonically sorted
before combination, so the result depends on the edge *set* and the spec —
never on the order rows came back from the database.

Cost is a single pass into a dict keyed by node pair, holding a fixed number
of running accumulators per pair. Memory scales with the number of distinct
pairs, not with the number of transactions, so a busy window collapses in
bounded space.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Where an edge's raw weight is read from.
WEIGHT_SOURCES = ("amount", "recency", "count")

#: How the raw weights of a repeated edge are combined into one.
COMBINERS = ("sum", "mean", "max", "min", "last")

_DEFAULT_HALF_LIFE_SECONDS = 7 * 86400

# Weights are rounded before they are emitted. Stellar amounts carry 7
# decimals; a couple of guard digits keep sums stable without inventing
# precision.
_PRECISION = 9


class InvalidWeightError(ValueError):
    """Raised when an edge cannot be weighted."""


@dataclass(frozen=True)
class WeightSpec:
    """How edge weights are derived and combined.

    Attributes:
        source: ``"amount"`` uses the transferred value, ``"recency"`` decays
            exponentially with the age of the transaction, ``"count"`` gives
            every transaction a weight of one (so ``sum`` counts them).
        combine: How repeated edges between the same pair are collapsed.
            ``"last"`` takes the value of the most recent transaction, which
            is the right choice for a recency weight and the wrong one for a
            volume weight — hence the explicit setting.
        half_life_seconds: For ``source="recency"``, the age at which an
            edge's weight halves.
        reference_timestamp: The "now" that ages are measured against. When
            omitted the newest timestamp in the snapshot is used, which keeps
            a rebuilt snapshot identical instead of drifting with wall clock.
        normalize: Scale the combined weights so the largest is 1.0. Useful
            when the weight feeds an attention or message-passing term that
            expects a bounded magnitude.
    """

    source: str = "amount"
    combine: str = "sum"
    half_life_seconds: float = _DEFAULT_HALF_LIFE_SECONDS
    reference_timestamp: int | None = None
    normalize: bool = False

    def validate(self) -> None:
        if self.source not in WEIGHT_SOURCES:
            raise ValueError(
                f"unknown weight source {self.source!r}; expected one of {WEIGHT_SOURCES}"
            )
        if self.combine not in COMBINERS:
            raise ValueError(f"unknown combiner {self.combine!r}; expected one of {COMBINERS}")
        if self.half_life_seconds <= 0:
            raise ValueError(f"half_life_seconds must be > 0, got {self.half_life_seconds}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "combine": self.combine,
            "half_life_seconds": self.half_life_seconds,
            "reference_timestamp": self.reference_timestamp,
            "normalize": self.normalize,
        }


@dataclass(frozen=True)
class WeightedEdge:
    """One node pair, with its combined weight and the evidence behind it."""

    src: str
    dst: str
    weight: float
    count: int
    first_timestamp: int | None
    last_timestamp: int | None

    def as_tuple(self) -> tuple[str, str, float]:
        return self.src, self.dst, self.weight


@dataclass(frozen=True)
class WeightedSnapshot:
    """A snapshot collapsed to one weighted edge per node pair."""

    spec: WeightSpec
    nodes: tuple[str, ...]
    edges: tuple[WeightedEdge, ...]

    def __len__(self) -> int:
        return len(self.edges)

    def weight_of(self, src: str, dst: str) -> float:
        """Weight of one pair, or 0.0 when the pair is not present."""
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                return edge.weight
        return 0.0

    def to_edge_index(self) -> tuple[list[list[int]], list[float]]:
        """``(edge_index, edge_weight)`` with node ids as row indices.

        ``edge_index`` is the ``[2, E]`` nested list torch_geometric expects
        and ``edge_weight`` the matching ``[E]`` list, ready to pass as
        ``edge_weight=`` to a GCNConv or as ``edge_attr``. Node order is
        :attr:`nodes`, so the two line up with any feature matrix built from
        the same snapshot.
        """
        index_of = {node: i for i, node in enumerate(self.nodes)}
        sources = [index_of[edge.src] for edge in self.edges]
        destinations = [index_of[edge.dst] for edge in self.edges]
        return [sources, destinations], [edge.weight for edge in self.edges]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "nodes": list(self.nodes),
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "weight": edge.weight,
                    "count": edge.count,
                    "first_timestamp": edge.first_timestamp,
                    "last_timestamp": edge.last_timestamp,
                }
                for edge in self.edges
            ],
        }


def _edge_parts(edge: Any) -> tuple[str, str, int | None, float | None]:
    """Extract (src, dst, timestamp, amount) from an Edge or a mapping."""
    if isinstance(edge, Mapping):
        raw_src = edge.get("src", edge.get("source"))
        raw_dst = edge.get("dst", edge.get("destination"))
        raw_ts = edge.get("timestamp")
        raw_amount = edge.get("amount")
    else:
        raw_src = getattr(edge, "src", None)
        raw_dst = getattr(edge, "dst", None)
        raw_ts = getattr(edge, "timestamp", None)
        raw_amount = getattr(edge, "amount", None)

    if raw_src is None or raw_dst is None:
        raise InvalidWeightError(f"edge is missing src or dst: {edge!r}")

    timestamp = None
    if raw_ts is not None:
        try:
            timestamp = int(raw_ts)
        except (TypeError, ValueError):
            raise InvalidWeightError(f"edge timestamp is not an integer: {edge!r}") from None

    amount = None
    if raw_amount is not None:
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            raise InvalidWeightError(f"edge amount is not numeric: {edge!r}") from None
        if math.isnan(amount) or math.isinf(amount):
            raise InvalidWeightError(f"edge amount must be finite: {edge!r}")
        if amount < 0:
            raise InvalidWeightError(f"edge amount must not be negative: {edge!r}")

    return str(raw_src), str(raw_dst), timestamp, amount


def _raw_weight(
    spec: WeightSpec,
    amount: float | None,
    timestamp: int | None,
    reference: int | None,
) -> float:
    """The weight of a single transaction, before any combining."""
    if spec.source == "count":
        return 1.0

    if spec.source == "amount":
        if amount is None:
            raise InvalidWeightError(
                "weight source is 'amount' but an edge carries no amount; "
                "use source='count' or supply amounts"
            )
        return amount

    # "recency": halve the weight for every half-life of age. Ages are
    # clamped at zero so an edge newer than the reference cannot score above
    # 1.0 and dominate everything else.
    if timestamp is None:
        raise InvalidWeightError("weight source is 'recency' but an edge carries no timestamp")
    if reference is None:
        return 1.0
    age = max(0, reference - timestamp)
    return float(2.0 ** (-age / spec.half_life_seconds))


class _PairAccumulator:
    """Running accumulators for one node pair.

    Deliberately not a list of weights: a pair seen a million times must cost
    the same as one seen twice.
    """

    __slots__ = ("total", "maximum", "minimum", "count", "first_ts", "last_ts", "last_weight")

    def __init__(self) -> None:
        self.total = 0.0
        self.maximum = -math.inf
        self.minimum = math.inf
        self.count = 0
        self.first_ts: int | None = None
        self.last_ts: int | None = None
        self.last_weight = 0.0

    def add(self, weight: float, timestamp: int | None) -> None:
        self.total += weight
        self.maximum = max(self.maximum, weight)
        self.minimum = min(self.minimum, weight)
        self.count += 1

        if timestamp is not None:
            if self.first_ts is None or timestamp < self.first_ts:
                self.first_ts = timestamp
            # >= so that, among transactions sharing the newest timestamp,
            # the last one in canonical order wins — a tie has to break the
            # same way every run.
            if self.last_ts is None or timestamp >= self.last_ts:
                self.last_ts = timestamp
                self.last_weight = weight
        elif self.count == 1 or self.last_ts is None:
            self.last_weight = weight

    def combined(self, combine: str) -> float:
        if combine == "sum":
            return self.total
        if combine == "mean":
            return self.total / self.count if self.count else 0.0
        if combine == "max":
            return self.maximum if self.count else 0.0
        if combine == "min":
            return self.minimum if self.count else 0.0
        return self.last_weight


def build_weighted_snapshot(
    edges: Iterable[Any],
    spec: WeightSpec | None = None,
) -> WeightedSnapshot:
    """Collapse ``edges`` into one weighted edge per node pair.

    Args:
        edges: Edge records — :class:`~astroml.features.graph.snapshot.Edge`
            instances or mappings with ``src``/``dst`` and, depending on the
            spec, ``amount`` and ``timestamp``.
        spec: How to weight and combine. Defaults to summed amounts.

    Returns:
        A :class:`WeightedSnapshot` with nodes and edges in sorted order.

    Raises:
        InvalidWeightError: if an edge is malformed, or lacks the field the
            chosen weight source needs.
    """
    spec = spec or WeightSpec()
    spec.validate()

    # Materialise and canonicalise first: the reference timestamp for a
    # recency weight has to be known before any weight is computed, and
    # sorting is what makes the combination order-independent.
    parsed = [_edge_parts(edge) for edge in edges]
    parsed.sort(key=lambda parts: (parts[0], parts[1], parts[2] is None, parts[2], parts[3]))

    reference = spec.reference_timestamp
    if spec.source == "recency" and reference is None:
        timestamps = [ts for _, _, ts, _ in parsed if ts is not None]
        reference = max(timestamps) if timestamps else None

    accumulators: dict[tuple[str, str], _PairAccumulator] = {}
    nodes: set[str] = set()

    for src, dst, timestamp, amount in parsed:
        nodes.add(src)
        nodes.add(dst)
        weight = _raw_weight(spec, amount, timestamp, reference)
        accumulators.setdefault((src, dst), _PairAccumulator()).add(weight, timestamp)

    combined = [
        WeightedEdge(
            src=src,
            dst=dst,
            weight=round(accumulator.combined(spec.combine), _PRECISION),
            count=accumulator.count,
            first_timestamp=accumulator.first_ts,
            last_timestamp=accumulator.last_ts,
        )
        for (src, dst), accumulator in sorted(accumulators.items())
    ]

    if spec.normalize and combined:
        largest = max(edge.weight for edge in combined)
        if largest > 0:
            combined = [
                WeightedEdge(
                    src=edge.src,
                    dst=edge.dst,
                    weight=round(edge.weight / largest, _PRECISION),
                    count=edge.count,
                    first_timestamp=edge.first_timestamp,
                    last_timestamp=edge.last_timestamp,
                )
                for edge in combined
            ]

    return WeightedSnapshot(spec=spec, nodes=tuple(sorted(nodes)), edges=tuple(combined))


__all__ = [
    "COMBINERS",
    "WEIGHT_SOURCES",
    "InvalidWeightError",
    "WeightSpec",
    "WeightedEdge",
    "WeightedSnapshot",
    "build_weighted_snapshot",
]
