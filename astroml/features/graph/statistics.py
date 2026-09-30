"""Per-snapshot graph statistics and JSON reporting — issue #740.

Every training snapshot gets a small, comparable summary: how big the graph
is, how dense, how the transferred amounts are distributed, and how the
edges are spread through the window. Written to a JSON report and
registered with the experiment tracker, these turn "the model got worse
this week" into a question you can answer by diffing two files.

The report is deterministic: the same edges in any order produce the same
document, and a ``generated_at`` stamp is deliberately *not* part of it, so
two runs over the same snapshot are byte-identical and can be compared with
a plain diff.

Statistics are computed in a single pass over the edges plus one sort of the
amounts, so the cost is O(E log E) with no intermediate graph object; a
window that does not fit in memory can be summarised with
:func:`SnapshotStatsAccumulator` instead of being materialised.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Report schema version. Bump when a field changes meaning so a consumer can
# tell an old report from a new one rather than silently mis-reading it.
REPORT_VERSION = "1.0.0"

_PERCENTILES = (50, 90, 95, 99)


class InvalidSnapshotError(ValueError):
    """Raised when a snapshot cannot be summarised."""


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile over an already sorted sequence.

    Implemented here rather than pulled from numpy so the reporting path has
    no array dependency and behaves identically on an empty window.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[int(rank)])
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


@dataclass(frozen=True)
class Distribution:
    """Summary of a numeric sample."""

    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    percentiles: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Distribution:
        if not values:
            return cls(percentiles={f"p{p}": 0.0 for p in _PERCENTILES})

        ordered = sorted(float(v) for v in values)
        count = len(ordered)
        total = math.fsum(ordered)
        mean = total / count
        # Population standard deviation: this describes the snapshot itself,
        # not a sample drawn from a wider population.
        variance = math.fsum((v - mean) ** 2 for v in ordered) / count
        return cls(
            count=count,
            total=round(total, 7),
            mean=round(mean, 7),
            std=round(math.sqrt(variance), 7),
            minimum=round(ordered[0], 7),
            maximum=round(ordered[-1], 7),
            percentiles={f"p{p}": round(_percentile(ordered, p), 7) for p in _PERCENTILES},
        )


@dataclass(frozen=True)
class TemporalSpread:
    """How the snapshot's edges are spread through its window."""

    first_timestamp: int | None = None
    last_timestamp: int | None = None
    span_seconds: int = 0
    mean_interarrival_seconds: float = 0.0
    busiest_second: int | None = None
    busiest_second_edges: int = 0


@dataclass(frozen=True)
class SnapshotStats:
    """Everything reported about one snapshot."""

    version: str
    index: int | None
    window_start: str | None
    window_end: str | None
    num_nodes: int
    num_edges: int
    num_unique_edges: int
    num_self_loops: int
    density: float
    reciprocity: float
    isolated_sinks: int
    isolated_sources: int
    degree: Distribution
    amounts: Distribution
    temporal: TemporalSpread

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def flat_metrics(self) -> dict[str, float]:
        """Scalar metrics suitable for an experiment tracker.

        Nested structures are flattened with dotted names because most
        trackers (MLflow included) only accept flat scalar metrics.
        """
        metrics: dict[str, float] = {
            "snapshot.num_nodes": float(self.num_nodes),
            "snapshot.num_edges": float(self.num_edges),
            "snapshot.num_unique_edges": float(self.num_unique_edges),
            "snapshot.num_self_loops": float(self.num_self_loops),
            "snapshot.density": float(self.density),
            "snapshot.reciprocity": float(self.reciprocity),
            "snapshot.isolated_sinks": float(self.isolated_sinks),
            "snapshot.isolated_sources": float(self.isolated_sources),
            "snapshot.degree.mean": float(self.degree.mean),
            "snapshot.degree.max": float(self.degree.maximum),
            "snapshot.amounts.count": float(self.amounts.count),
            "snapshot.amounts.total": float(self.amounts.total),
            "snapshot.amounts.mean": float(self.amounts.mean),
            "snapshot.amounts.std": float(self.amounts.std),
            "snapshot.temporal.span_seconds": float(self.temporal.span_seconds),
            "snapshot.temporal.mean_interarrival_seconds": float(
                self.temporal.mean_interarrival_seconds
            ),
        }
        for name, value in self.amounts.percentiles.items():
            metrics[f"snapshot.amounts.{name}"] = float(value)
        return metrics


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


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
        raise InvalidSnapshotError(f"edge is missing src or dst: {edge!r}")

    timestamp = None
    if raw_ts is not None:
        try:
            timestamp = int(raw_ts)
        except (TypeError, ValueError):
            raise InvalidSnapshotError(f"edge timestamp is not an integer: {edge!r}") from None

    amount = None
    if raw_amount is not None:
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            raise InvalidSnapshotError(f"edge amount is not numeric: {edge!r}") from None
        if math.isnan(amount) or math.isinf(amount):
            raise InvalidSnapshotError(f"edge amount must be finite: {edge!r}")

    return str(raw_src), str(raw_dst), timestamp, amount


def compute_snapshot_stats(
    edges: Iterable[Any],
    index: int | None = None,
    window_start: datetime | str | None = None,
    window_end: datetime | str | None = None,
    nodes: Iterable[str] | None = None,
) -> SnapshotStats:
    """Summarise one snapshot.

    Args:
        edges: Edge records — ``Edge`` instances or mappings.
        index: Snapshot index within the series, if it has one.
        window_start / window_end: Window bounds, recorded verbatim.
        nodes: Optional node set. When given it is used as the vertex set,
            so a node with no edges in this window still counts towards
            density. When omitted the vertex set is inferred from the edges.

    Returns:
        A :class:`SnapshotStats`.

    Raises:
        InvalidSnapshotError: if an edge is malformed.
    """
    accumulator = SnapshotStatsAccumulator(
        index=index, window_start=window_start, window_end=window_end
    )
    for edge in edges:
        accumulator.add(edge)
    if nodes is not None:
        accumulator.declare_nodes(nodes)
    return accumulator.result()


class SnapshotStatsAccumulator:
    """Streaming counterpart to :func:`compute_snapshot_stats`.

    Feed edges one at a time — from :func:`iter_db_snapshot_edges`, for
    instance — and the accumulator keeps only per-node counters and the
    amount sample, never the edge list. That keeps peak memory proportional
    to the number of distinct accounts rather than to the number of
    transactions, which is the difference that matters on a busy window.
    """

    def __init__(
        self,
        index: int | None = None,
        window_start: datetime | str | None = None,
        window_end: datetime | str | None = None,
    ) -> None:
        self.index = index
        self.window_start = window_start
        self.window_end = window_end

        self._out_degree: dict[str, int] = {}
        self._in_degree: dict[str, int] = {}
        self._nodes: set[str] = set()
        self._pairs: set[tuple[str, str]] = set()
        self._amounts: list[float] = []
        self._timestamps: list[int] = []
        self._per_second: dict[int, int] = {}
        self._num_edges = 0
        self._self_loops = 0

    def declare_nodes(self, nodes: Iterable[str]) -> None:
        """Add nodes that may not appear on any edge in this window."""
        for node in nodes:
            self._nodes.add(str(node))

    def add(self, edge: Any) -> None:
        src, dst, timestamp, amount = _edge_parts(edge)

        self._num_edges += 1
        self._nodes.add(src)
        self._nodes.add(dst)
        self._out_degree[src] = self._out_degree.get(src, 0) + 1
        self._in_degree[dst] = self._in_degree.get(dst, 0) + 1
        self._pairs.add((src, dst))
        if src == dst:
            self._self_loops += 1
        if amount is not None:
            self._amounts.append(amount)
        if timestamp is not None:
            self._timestamps.append(timestamp)
            self._per_second[timestamp] = self._per_second.get(timestamp, 0) + 1

    def result(self) -> SnapshotStats:
        node_count = len(self._nodes)
        unique_edges = len(self._pairs)

        # Directed density over ordered pairs, self-loops excluded from the
        # denominator: n*(n-1) possible directed edges.
        possible = node_count * (node_count - 1)
        density = (unique_edges / possible) if possible else 0.0

        reciprocal = sum(1 for (a, b) in self._pairs if a != b and (b, a) in self._pairs)
        non_loop_pairs = sum(1 for (a, b) in self._pairs if a != b)
        reciprocity = (reciprocal / non_loop_pairs) if non_loop_pairs else 0.0

        degrees = [
            self._out_degree.get(node, 0) + self._in_degree.get(node, 0)
            for node in sorted(self._nodes)
        ]

        temporal = TemporalSpread()
        if self._timestamps:
            ordered_ts = sorted(self._timestamps)
            span = ordered_ts[-1] - ordered_ts[0]
            gaps = len(ordered_ts) - 1
            busiest_second = max(sorted(self._per_second), key=lambda s: self._per_second[s])
            temporal = TemporalSpread(
                first_timestamp=ordered_ts[0],
                last_timestamp=ordered_ts[-1],
                span_seconds=span,
                mean_interarrival_seconds=round(span / gaps, 7) if gaps else 0.0,
                busiest_second=busiest_second,
                busiest_second_edges=self._per_second[busiest_second],
            )

        return SnapshotStats(
            version=REPORT_VERSION,
            index=self.index,
            window_start=_iso(self.window_start),
            window_end=_iso(self.window_end),
            num_nodes=node_count,
            num_edges=self._num_edges,
            num_unique_edges=unique_edges,
            num_self_loops=self._self_loops,
            density=round(density, 9),
            reciprocity=round(reciprocity, 9),
            isolated_sinks=sum(1 for node in self._nodes if not self._out_degree.get(node)),
            isolated_sources=sum(1 for node in self._nodes if not self._in_degree.get(node)),
            degree=Distribution.from_values(degrees),
            amounts=Distribution.from_values(self._amounts),
            temporal=temporal,
        )


def build_snapshot_report(
    stats: Sequence[SnapshotStats],
    name: str = "snapshot-report",
) -> dict[str, Any]:
    """Assemble a report document covering a series of snapshots."""
    documents = [stat.to_dict() for stat in stats]
    return {
        "version": REPORT_VERSION,
        "name": name,
        "num_snapshots": len(documents),
        "totals": {
            "edges": sum(stat.num_edges for stat in stats),
            "nodes": sum(stat.num_nodes for stat in stats),
        },
        "snapshots": documents,
    }


def write_snapshot_report(
    stats: Sequence[SnapshotStats],
    path: str | Path,
    name: str = "snapshot-report",
) -> Path:
    """Write the report to ``path`` as JSON and return the path."""
    report = build_snapshot_report(stats, name=name)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a reader picking the file up mid-run never sees a
    # partial document.
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(destination)
    return destination


def register_snapshot_report(
    stats: Sequence[SnapshotStats],
    tracker: Any,
    path: str | Path,
    name: str = "snapshot-report",
    artifact_path: str = "snapshots",
) -> Path:
    """Write the report and register it with an experiment tracker.

    ``tracker`` is anything exposing MLflow-style ``save_artifact`` and
    ``log_metrics`` (:class:`astroml.tracking.mlflow_tracker.MLflowTracker`
    does). Per-snapshot scalars are logged with the snapshot index as the
    step so they plot as a series; a tracker that is disabled or missing a
    method is skipped rather than raising, because a reporting failure must
    not take down a training run.

    Returns the path the report was written to.
    """
    destination = write_snapshot_report(stats, path, name=name)

    log_metrics = getattr(tracker, "log_metrics", None)
    if callable(log_metrics):
        for step, stat in enumerate(stats):
            log_metrics(stat.flat_metrics(), step=stat.index if stat.index is not None else step)

    save_artifact = getattr(tracker, "save_artifact", None)
    if callable(save_artifact):
        save_artifact(destination, artifact_path=artifact_path)

    return destination


__all__ = [
    "REPORT_VERSION",
    "Distribution",
    "InvalidSnapshotError",
    "SnapshotStats",
    "SnapshotStatsAccumulator",
    "TemporalSpread",
    "build_snapshot_report",
    "compute_snapshot_stats",
    "register_snapshot_report",
    "write_snapshot_report",
]
