"""Per-snapshot statistics and JSON reporting (issue #740).

The report only earns its keep if two runs over the same snapshot produce
the same document — otherwise every diff is noise. These tests pin that,
the arithmetic behind each field, and the tracker registration.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

import pytest

from astroml.features.graph.statistics import (
    REPORT_VERSION,
    Distribution,
    InvalidSnapshotError,
    SnapshotStatsAccumulator,
    build_snapshot_report,
    compute_snapshot_stats,
    register_snapshot_report,
    write_snapshot_report,
)

TRIANGLE = [
    {"src": "a", "dst": "b", "timestamp": 100, "amount": 10.0},
    {"src": "b", "dst": "c", "timestamp": 110, "amount": 20.0},
    {"src": "c", "dst": "a", "timestamp": 130, "amount": 30.0},
]


class TestGraphShape:
    def test_nodes_and_edges_are_counted(self):
        stats = compute_snapshot_stats(TRIANGLE)

        assert stats.num_nodes == 3
        assert stats.num_edges == 3

    def test_parallel_edges_count_once_towards_unique_edges(self):
        stats = compute_snapshot_stats(
            [
                {"src": "a", "dst": "b", "timestamp": 1},
                {"src": "a", "dst": "b", "timestamp": 2},
            ]
        )

        assert stats.num_edges == 2
        assert stats.num_unique_edges == 1

    def test_self_loops_are_reported(self):
        stats = compute_snapshot_stats(
            [{"src": "a", "dst": "a", "timestamp": 1}, {"src": "a", "dst": "b", "timestamp": 2}]
        )

        assert stats.num_self_loops == 1

    def test_density_is_over_ordered_pairs(self):
        # 3 nodes -> 6 possible directed edges; the triangle uses 3.
        stats = compute_snapshot_stats(TRIANGLE)

        assert stats.density == pytest.approx(0.5)

    def test_density_of_a_single_node_graph_is_zero_not_a_division_error(self):
        stats = compute_snapshot_stats([{"src": "a", "dst": "a", "timestamp": 1}])

        assert stats.density == 0.0

    def test_reciprocity_counts_edges_answered_in_both_directions(self):
        stats = compute_snapshot_stats(
            [
                {"src": "a", "dst": "b", "timestamp": 1},
                {"src": "b", "dst": "a", "timestamp": 2},
                {"src": "a", "dst": "c", "timestamp": 3},
            ]
        )

        # 2 of the 3 directed pairs have a mirror.
        assert stats.reciprocity == pytest.approx(2 / 3)

    def test_declared_nodes_widen_the_vertex_set(self):
        stats = compute_snapshot_stats(TRIANGLE, nodes=["a", "b", "c", "d", "e"])

        # Accounts present in the window but transacting with nobody still
        # dilute the density, and the report should say so.
        assert stats.num_nodes == 5
        assert stats.density < 0.5

    def test_sources_and_sinks_are_identified(self):
        stats = compute_snapshot_stats([{"src": "a", "dst": "b", "timestamp": 1}])

        assert stats.isolated_sinks == 1  # b never sends
        assert stats.isolated_sources == 1  # a never receives

    def test_an_empty_snapshot_reports_zeroes(self):
        stats = compute_snapshot_stats([], index=3)

        assert stats.num_nodes == 0
        assert stats.num_edges == 0
        assert stats.density == 0.0
        assert stats.amounts.count == 0
        assert stats.temporal.first_timestamp is None


class TestAmountDistribution:
    def test_summary_statistics_are_computed(self):
        stats = compute_snapshot_stats(TRIANGLE)

        assert stats.amounts.count == 3
        assert stats.amounts.total == pytest.approx(60.0)
        assert stats.amounts.mean == pytest.approx(20.0)
        assert stats.amounts.minimum == pytest.approx(10.0)
        assert stats.amounts.maximum == pytest.approx(30.0)

    def test_percentiles_are_reported(self):
        stats = compute_snapshot_stats(TRIANGLE)

        assert stats.amounts.percentiles["p50"] == pytest.approx(20.0)
        assert stats.amounts.percentiles["p99"] == pytest.approx(30.0, abs=0.5)

    def test_percentiles_are_monotonic(self):
        values = [float(v) for v in range(1, 101)]
        distribution = Distribution.from_values(values)

        ordered = [distribution.percentiles[f"p{p}"] for p in (50, 90, 95, 99)]
        assert ordered == sorted(ordered)

    def test_a_single_value_has_zero_spread(self):
        distribution = Distribution.from_values([7.0])

        assert distribution.mean == pytest.approx(7.0)
        assert distribution.std == 0.0
        assert distribution.percentiles["p90"] == pytest.approx(7.0)

    def test_edges_without_amounts_produce_an_empty_distribution(self):
        stats = compute_snapshot_stats([{"src": "a", "dst": "b", "timestamp": 1}])

        assert stats.amounts.count == 0
        assert stats.amounts.mean == 0.0


class TestTemporalSpread:
    def test_bounds_and_span_are_reported(self):
        stats = compute_snapshot_stats(TRIANGLE)

        assert stats.temporal.first_timestamp == 100
        assert stats.temporal.last_timestamp == 130
        assert stats.temporal.span_seconds == 30

    def test_mean_interarrival_divides_by_the_gaps_not_the_edges(self):
        stats = compute_snapshot_stats(TRIANGLE)

        # Three edges, two gaps, 30 seconds of span.
        assert stats.temporal.mean_interarrival_seconds == pytest.approx(15.0)

    def test_a_single_edge_has_no_interarrival(self):
        stats = compute_snapshot_stats([{"src": "a", "dst": "b", "timestamp": 5}])

        assert stats.temporal.span_seconds == 0
        assert stats.temporal.mean_interarrival_seconds == 0.0

    def test_the_busiest_second_is_identified(self):
        stats = compute_snapshot_stats(
            [
                {"src": "a", "dst": "b", "timestamp": 10},
                {"src": "a", "dst": "c", "timestamp": 20},
                {"src": "a", "dst": "d", "timestamp": 20},
            ]
        )

        assert stats.temporal.busiest_second == 20
        assert stats.temporal.busiest_second_edges == 2


class TestDeterminism:
    def test_edge_order_does_not_change_the_report(self):
        shuffled = list(TRIANGLE)
        random.Random(99).shuffle(shuffled)

        assert compute_snapshot_stats(TRIANGLE) == compute_snapshot_stats(shuffled)

    def test_the_json_document_is_byte_identical_across_runs(self):
        first = compute_snapshot_stats(TRIANGLE, index=0).to_json()
        second = compute_snapshot_stats(list(reversed(TRIANGLE)), index=0).to_json()

        # No wall-clock stamp in the document: two runs over one snapshot
        # must diff clean.
        assert first == second

    def test_the_report_carries_its_schema_version(self):
        assert compute_snapshot_stats(TRIANGLE).version == REPORT_VERSION


class TestStreamingAccumulator:
    def test_streaming_matches_the_batch_computation(self):
        accumulator = SnapshotStatsAccumulator(index=2)
        for edge in TRIANGLE:
            accumulator.add(edge)

        assert accumulator.result() == compute_snapshot_stats(TRIANGLE, index=2)

    def test_it_never_retains_the_edge_list(self):
        accumulator = SnapshotStatsAccumulator()
        for i in range(1000):
            accumulator.add({"src": f"n{i % 10}", "dst": f"n{(i + 1) % 10}", "timestamp": i})

        # Per-node counters only: 10 accounts, not 1000 edges.
        assert accumulator.result().num_nodes == 10
        assert accumulator.result().num_edges == 1000


class TestValidation:
    def test_an_edge_without_endpoints_is_rejected(self):
        with pytest.raises(InvalidSnapshotError, match="missing src or dst"):
            compute_snapshot_stats([{"src": "a"}])

    def test_a_non_numeric_amount_is_rejected(self):
        with pytest.raises(InvalidSnapshotError, match="not numeric"):
            compute_snapshot_stats([{"src": "a", "dst": "b", "amount": "many"}])

    def test_an_infinite_amount_is_rejected(self):
        with pytest.raises(InvalidSnapshotError, match="finite"):
            compute_snapshot_stats([{"src": "a", "dst": "b", "amount": float("inf")}])

    def test_a_non_integer_timestamp_is_rejected(self):
        with pytest.raises(InvalidSnapshotError, match="timestamp"):
            compute_snapshot_stats([{"src": "a", "dst": "b", "timestamp": "yesterday"}])


class TestSnapshotEdges:
    def test_snapshot_edge_dataclasses_are_accepted(self):
        from astroml.features.graph.snapshot import Edge

        stats = compute_snapshot_stats([Edge(src="a", dst="b", timestamp=1, amount=9.0)])

        assert stats.num_edges == 1
        assert stats.amounts.total == pytest.approx(9.0)

    def test_window_bounds_are_recorded(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 8, tzinfo=timezone.utc)

        stats = compute_snapshot_stats(TRIANGLE, index=4, window_start=start, window_end=end)

        assert stats.index == 4
        assert stats.window_start == start.isoformat()
        assert stats.window_end == end.isoformat()


class TestReportOutput:
    def test_the_report_covers_every_snapshot(self):
        stats = [compute_snapshot_stats(TRIANGLE, index=i) for i in range(3)]

        report = build_snapshot_report(stats, name="weekly")

        assert report["name"] == "weekly"
        assert report["num_snapshots"] == 3
        assert report["totals"]["edges"] == 9

    def test_it_is_written_as_readable_json(self, tmp_path):
        stats = [compute_snapshot_stats(TRIANGLE, index=0)]

        path = write_snapshot_report(stats, tmp_path / "reports" / "run.json")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["snapshots"][0]["num_edges"] == 3

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = write_snapshot_report([], tmp_path / "deep" / "nested" / "run.json")

        assert path.is_file()

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        write_snapshot_report([compute_snapshot_stats(TRIANGLE)], tmp_path / "run.json")

        assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


class _RecordingTracker:
    """Stands in for MLflowTracker; records what it was asked to log."""

    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.artifacts: list[tuple[str, str]] = []

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))

    def save_artifact(self, local_path, artifact_path="artifacts"):
        self.artifacts.append((str(local_path), artifact_path))
        return f"memory://{artifact_path}"


class TestExperimentRegistration:
    def test_the_report_is_registered_as_an_artifact(self, tmp_path):
        tracker = _RecordingTracker()
        stats = [compute_snapshot_stats(TRIANGLE, index=0)]

        path = register_snapshot_report(stats, tracker, tmp_path / "run.json")

        assert tracker.artifacts == [(str(path), "snapshots")]

    def test_scalar_metrics_are_logged_per_snapshot(self, tmp_path):
        tracker = _RecordingTracker()
        stats = [compute_snapshot_stats(TRIANGLE, index=i) for i in range(2)]

        register_snapshot_report(stats, tracker, tmp_path / "run.json")

        assert [step for _, step in tracker.metrics] == [0, 1]
        assert tracker.metrics[0][0]["snapshot.num_edges"] == 3.0

    def test_logged_metrics_are_all_flat_scalars(self, tmp_path):
        tracker = _RecordingTracker()

        register_snapshot_report(
            [compute_snapshot_stats(TRIANGLE, index=0)], tracker, tmp_path / "run.json"
        )

        # MLflow rejects anything but a flat name -> float mapping.
        metrics, _ = tracker.metrics[0]
        assert all(isinstance(value, float) for value in metrics.values())
        assert "snapshot.amounts.p90" in metrics

    def test_a_tracker_that_cannot_log_does_not_break_the_run(self, tmp_path):
        class _Disabled:
            pass

        # A reporting failure must never take a training run down with it.
        path = register_snapshot_report(
            [compute_snapshot_stats(TRIANGLE)], _Disabled(), tmp_path / "run.json"
        )

        assert path.is_file()
