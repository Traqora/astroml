"""Neighbourhood feature precomputation for transaction graphs — issue #735.

Cheap, structural features that can be handed straight to a GNN as input
without a forward pass: degrees, per-node amount statistics, hop-N
neighbourhood sizes, and aggregates over a node's immediate neighbours.

Three properties matter here and are enforced rather than assumed:

*Deterministic* — nodes are emitted in sorted order, feature columns in a
fixed order, and every accumulation runs over a sorted adjacency, so the
same edge list always produces identical output regardless of the order the
edges arrived in.

*Validated* — the edge list is normalised and checked up front
(:class:`InvalidGraphError`), so a malformed row fails at the boundary
rather than silently contributing a NaN to a training tensor.

*Cheap on large graphs* — adjacency is built in a single pass, hop
expansion is a breadth-first frontier that visits each node at most once
per hop, and a per-node budget bounds the blow-up on hub nodes. Results are
content-addressed and cached, so a repeated run over an unchanged snapshot
costs a hash of the edge list.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bump when the meaning or set of emitted features changes. The version is
# part of every cache key, so a bump invalidates previously cached vectors
# instead of silently mixing two feature generations in one training run.
FEATURE_VERSION = "1.0.0"

# Amounts are rounded before they are summed and again before they are
# emitted. Stellar amounts carry 7 decimal places; keeping a couple of guard
# digits keeps totals stable without inventing precision.
_AMOUNT_PRECISION = 9


class InvalidGraphError(ValueError):
    """Raised when an edge list cannot be interpreted as a graph."""


@dataclass(frozen=True)
class NeighborhoodConfig:
    """Which neighbourhood features to precompute.

    Attributes:
        max_hops: Highest ``hop{k}_size`` to emit. 1 gives immediate
            neighbours only; each additional hop costs another sweep of the
            frontier.
        directed: When true, hop expansion follows edge direction
            (out-neighbours only). When false an edge is traversable both
            ways, which is usually what you want for reachability but makes
            hub nodes considerably more expensive.
        neighbour_aggregates: Emit means over a node's immediate neighbours
            (their degree, their average amount). This is the
            neighbour-based signal a GNN would otherwise spend its first
            layer learning.
        hop_node_budget: Stop expanding a node's frontier once this many
            nodes have been reached. Bounds the worst case on a graph with a
            few very high-degree accounts, at the cost of a truncated —
            never overstated — hop count. ``None`` disables the bound.
    """

    max_hops: int = 2
    directed: bool = True
    neighbour_aggregates: bool = True
    hop_node_budget: int | None = 10_000

    def validate(self) -> None:
        if self.max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {self.max_hops}")
        if self.hop_node_budget is not None and self.hop_node_budget < 1:
            raise ValueError(f"hop_node_budget must be >= 1 or None, got {self.hop_node_budget}")

    def fingerprint(self) -> str:
        """Stable hash of the settings that affect the emitted values."""
        payload = json.dumps(
            {
                "max_hops": self.max_hops,
                "directed": self.directed,
                "neighbour_aggregates": self.neighbour_aggregates,
                "hop_node_budget": self.hop_node_budget,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class NeighborhoodFeatures:
    """Precomputed features for every node in a graph.

    ``nodes`` and ``feature_names`` are both fixed orderings, so
    :meth:`to_matrix` yields a row and column layout that is reproducible
    across processes and runs.
    """

    version: str
    config_fingerprint: str
    graph_fingerprint: str
    nodes: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: dict[str, dict[str, float]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def for_node(self, node: str) -> dict[str, float]:
        try:
            return dict(self.values[node])
        except KeyError:
            raise KeyError(f"node {node!r} is not in this graph") from None

    def to_matrix(self) -> tuple[tuple[str, ...], list[list[float]]]:
        """Return ``(node_ids, rows)`` in the canonical ordering."""
        rows = [[self.values[node][name] for name in self.feature_names] for node in self.nodes]
        return self.nodes, rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_fingerprint": self.config_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "nodes": list(self.nodes),
            "feature_names": list(self.feature_names),
            "values": {node: dict(cols) for node, cols in self.values.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NeighborhoodFeatures:
        return cls(
            version=payload["version"],
            config_fingerprint=payload["config_fingerprint"],
            graph_fingerprint=payload["graph_fingerprint"],
            nodes=tuple(payload["nodes"]),
            feature_names=tuple(payload["feature_names"]),
            values={node: dict(cols) for node, cols in payload["values"].items()},
        )


# ---------------------------------------------------------------------------
# Edge normalisation and validation
# ---------------------------------------------------------------------------


def _edge_fields(edge: Any) -> tuple[str, str, float]:
    """Pull (src, dst, amount) out of a dataclass Edge or a mapping."""
    if isinstance(edge, Mapping):
        raw_src = edge.get("src", edge.get("source", edge.get("source_account")))
        raw_dst = edge.get("dst", edge.get("destination", edge.get("destination_account")))
        raw_amount = edge.get("amount", 0.0)
    else:
        raw_src = getattr(edge, "src", None)
        raw_dst = getattr(edge, "dst", None)
        raw_amount = getattr(edge, "amount", 0.0)

    if raw_src is None or raw_dst is None:
        raise InvalidGraphError(f"edge is missing src or dst: {edge!r}")

    src, dst = str(raw_src), str(raw_dst)
    if not src or not dst:
        raise InvalidGraphError(f"edge has an empty endpoint: {edge!r}")

    if raw_amount is None:
        amount = 0.0
    else:
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            raise InvalidGraphError(f"edge amount is not numeric: {edge!r}") from None
        if math.isnan(amount) or math.isinf(amount):
            raise InvalidGraphError(f"edge amount must be finite: {edge!r}")
        if amount < 0:
            raise InvalidGraphError(f"edge amount must not be negative: {edge!r}")

    return src, dst, round(amount, _AMOUNT_PRECISION)


def normalize_edges(edges: Iterable[Any]) -> list[tuple[str, str, float]]:
    """Validate and canonicalise an edge list.

    Accepts :class:`astroml.features.graph.snapshot.Edge` instances or plain
    mappings. The result is sorted, which is what makes every downstream
    accumulation order-independent.
    """
    normalized = [_edge_fields(edge) for edge in edges]
    normalized.sort()
    return normalized


def graph_fingerprint(edges: Sequence[tuple[str, str, float]]) -> str:
    """Content hash of a normalised edge list.

    Two graphs with the same edges in a different order hash the same, so a
    reshuffled snapshot is a cache hit rather than a recomputation.
    """
    digest = hashlib.sha256()
    for src, dst, amount in edges:
        digest.update(f"{src}\x1f{dst}\x1f{amount:.{_AMOUNT_PRECISION}f}\x1e".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class NeighborhoodFeatureCache:
    """Versioned cache for precomputed neighbourhood features.

    Keys combine :data:`FEATURE_VERSION`, the config fingerprint and the
    graph fingerprint, so a feature-set change, a config change or an edge
    change each produce a distinct entry and none of them can be served a
    stale vector.

    With no ``directory`` the cache is process-local; with one, entries are
    JSON files that survive restarts. Deliberately not Redis-backed: this
    runs inside the training loop, where a missing server should mean a
    recomputation, not a failure.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._memory: dict[str, NeighborhoodFeatures] = {}
        self._directory = Path(directory) if directory is not None else None
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(config_fingerprint: str, graph_fp: str) -> str:
        return f"nbhd-v{FEATURE_VERSION}-{config_fingerprint}-{graph_fp[:32]}"

    def _path(self, key: str) -> Path | None:
        return None if self._directory is None else self._directory / f"{key}.json"

    def get(self, key: str) -> NeighborhoodFeatures | None:
        hit = self._memory.get(key)
        if hit is not None:
            return hit

        path = self._path(key)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = NeighborhoodFeatures.from_dict(payload)
        except (OSError, ValueError, KeyError):
            # A truncated or hand-edited cache file is not worth failing a
            # training run over — treat it as a miss and let it be rewritten.
            return None

        if features.version != FEATURE_VERSION:
            return None
        self._memory[key] = features
        return features

    def put(self, key: str, features: NeighborhoodFeatures) -> None:
        self._memory[key] = features
        path = self._path(key)
        if path is None:
            return
        # Write-then-rename so a crash mid-write cannot leave a half-written
        # file that a later run would have to guess about.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(features.to_dict(), sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def clear(self) -> None:
        self._memory.clear()
        if self._directory is not None:
            for path in self._directory.glob("nbhd-v*.json"):
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------


def _base_feature_names(config: NeighborhoodConfig) -> tuple[str, ...]:
    names = [
        "out_degree",
        "in_degree",
        "degree",
        "unique_out_neighbors",
        "unique_in_neighbors",
        "total_sent",
        "total_received",
        "mean_sent_amount",
        "mean_received_amount",
        "max_sent_amount",
    ]
    names.extend(f"hop{k}_size" for k in range(1, config.max_hops + 1))
    if config.neighbour_aggregates:
        names.extend(["neighbour_mean_degree", "neighbour_mean_amount"])
    return tuple(names)


def _hop_sizes(
    start: str,
    adjacency: Mapping[str, tuple[str, ...]],
    max_hops: int,
    budget: int | None,
) -> list[int]:
    """Breadth-first frontier expansion, one entry per hop.

    Each node is visited at most once across the whole walk, so the cost is
    bounded by the size of the reachable set rather than by the number of
    paths — the difference between linear and combinatorial on a dense
    transaction graph.
    """
    seen = {start}
    frontier = [start]
    sizes: list[int] = []
    truncated = False

    for _ in range(max_hops):
        next_frontier: list[str] = []
        for node in frontier:
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    next_frontier.append(neighbour)
            if budget is not None and len(seen) > budget:
                truncated = True
                break
        frontier = next_frontier
        # The start node is excluded from its own reachable count.
        sizes.append(len(seen) - 1)
        if truncated or not frontier:
            # Remaining hops would add nothing (or cannot be trusted once
            # truncated); carry the last count forward so the sequence stays
            # non-decreasing.
            sizes.extend([len(seen) - 1] * (max_hops - len(sizes)))
            break

    return sizes[:max_hops]


def precompute_neighborhood_features(
    edges: Iterable[Any],
    config: NeighborhoodConfig | None = None,
    cache: NeighborhoodFeatureCache | None = None,
) -> NeighborhoodFeatures:
    """Precompute neighbourhood features for every node in ``edges``.

    Args:
        edges: Edge records — ``Edge`` instances or mappings carrying
            ``src``/``dst`` and an optional ``amount``.
        config: Which features to emit. Defaults to
            :class:`NeighborhoodConfig`.
        cache: Optional versioned cache. On a hit the edge list is only
            hashed, not walked.

    Returns:
        A :class:`NeighborhoodFeatures` with nodes and columns in a fixed,
        reproducible order.

    Raises:
        InvalidGraphError: if an edge is malformed.
    """
    config = config or NeighborhoodConfig()
    config.validate()

    normalized = normalize_edges(edges)
    graph_fp = graph_fingerprint(normalized)
    config_fp = config.fingerprint()

    key = NeighborhoodFeatureCache.make_key(config_fp, graph_fp)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    out_count: dict[str, int] = {}
    in_count: dict[str, int] = {}
    out_neighbours: dict[str, set[str]] = {}
    in_neighbours: dict[str, set[str]] = {}
    sent: dict[str, float] = {}
    received: dict[str, float] = {}
    max_sent: dict[str, float] = {}
    nodes: set[str] = set()

    # Single pass: everything above is accumulated together so a large edge
    # list is streamed once rather than re-scanned per feature.
    for src, dst, amount in normalized:
        nodes.add(src)
        nodes.add(dst)
        out_count[src] = out_count.get(src, 0) + 1
        in_count[dst] = in_count.get(dst, 0) + 1
        out_neighbours.setdefault(src, set()).add(dst)
        in_neighbours.setdefault(dst, set()).add(src)
        sent[src] = sent.get(src, 0.0) + amount
        received[dst] = received.get(dst, 0.0) + amount
        if amount > max_sent.get(src, 0.0):
            max_sent[src] = amount

    ordered_nodes = tuple(sorted(nodes))

    # Sorted adjacency tuples: fixed iteration order for the hop walk, and
    # cheaper to traverse repeatedly than the sets they came from.
    if config.directed:
        adjacency = {node: tuple(sorted(peers)) for node, peers in out_neighbours.items()}
    else:
        adjacency = {
            node: tuple(sorted(out_neighbours.get(node, set()) | in_neighbours.get(node, set())))
            for node in ordered_nodes
        }

    feature_names = _base_feature_names(config)
    values: dict[str, dict[str, float]] = {}

    degree_of = {node: out_count.get(node, 0) + in_count.get(node, 0) for node in ordered_nodes}
    mean_sent_of = {
        node: (sent.get(node, 0.0) / out_count[node]) if out_count.get(node) else 0.0
        for node in ordered_nodes
    }

    for node in ordered_nodes:
        out_deg = out_count.get(node, 0)
        in_deg = in_count.get(node, 0)
        row: dict[str, float] = {
            "out_degree": float(out_deg),
            "in_degree": float(in_deg),
            "degree": float(out_deg + in_deg),
            "unique_out_neighbors": float(len(out_neighbours.get(node, ()))),
            "unique_in_neighbors": float(len(in_neighbours.get(node, ()))),
            "total_sent": round(sent.get(node, 0.0), _AMOUNT_PRECISION),
            "total_received": round(received.get(node, 0.0), _AMOUNT_PRECISION),
            "mean_sent_amount": round(mean_sent_of[node], _AMOUNT_PRECISION),
            "mean_received_amount": round(
                (received.get(node, 0.0) / in_deg) if in_deg else 0.0, _AMOUNT_PRECISION
            ),
            "max_sent_amount": round(max_sent.get(node, 0.0), _AMOUNT_PRECISION),
        }

        for index, size in enumerate(
            _hop_sizes(node, adjacency, config.max_hops, config.hop_node_budget), start=1
        ):
            row[f"hop{index}_size"] = float(size)

        if config.neighbour_aggregates:
            peers = adjacency.get(node, ())
            if peers:
                row["neighbour_mean_degree"] = round(
                    sum(degree_of.get(peer, 0) for peer in peers) / len(peers),
                    _AMOUNT_PRECISION,
                )
                row["neighbour_mean_amount"] = round(
                    sum(mean_sent_of.get(peer, 0.0) for peer in peers) / len(peers),
                    _AMOUNT_PRECISION,
                )
            else:
                row["neighbour_mean_degree"] = 0.0
                row["neighbour_mean_amount"] = 0.0

        values[node] = row

    features = NeighborhoodFeatures(
        version=FEATURE_VERSION,
        config_fingerprint=config_fp,
        graph_fingerprint=graph_fp,
        nodes=ordered_nodes,
        feature_names=feature_names,
        values=values,
    )

    if cache is not None:
        cache.put(key, features)
    return features


__all__ = [
    "FEATURE_VERSION",
    "InvalidGraphError",
    "NeighborhoodConfig",
    "NeighborhoodFeatureCache",
    "NeighborhoodFeatures",
    "graph_fingerprint",
    "normalize_edges",
    "precompute_neighborhood_features",
]
