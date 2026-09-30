"""Weighted edges in the snapshot builder (issue #731).

The interesting behaviour is how repeated edges collapse: the combiners have
to disagree in the ways they claim to, ties have to break the same way every
run, and the result must not depend on the order rows came back from the
database.
"""

from __future__ import annotations

import random

import pytest

from astroml.features.graph.weights import (
    COMBINERS,
    WEIGHT_SOURCES,
    InvalidWeightError,
    WeightedSnapshot,
    WeightSpec,
    build_weighted_snapshot,
)

# a -> b twice (10 then 30), b -> c once.
REPEATED = [
    {"src": "a", "dst": "b", "amount": 10.0, "timestamp": 100},
    {"src": "a", "dst": "b", "amount": 30.0, "timestamp": 200},
    {"src": "b", "dst": "c", "amount": 5.0, "timestamp": 150},
]


def _weight(snapshot: WeightedSnapshot, src: str, dst: str) -> float:
    return snapshot.weight_of(src, dst)


class TestCombiners:
    @pytest.mark.parametrize(
        "combine, expected",
        [("sum", 40.0), ("mean", 20.0), ("max", 30.0), ("min", 10.0), ("last", 30.0)],
    )
    def test_repeated_edges_collapse_as_configured(self, combine, expected):
        snapshot = build_weighted_snapshot(REPEATED, WeightSpec(combine=combine))

        assert _weight(snapshot, "a", "b") == pytest.approx(expected)

    def test_every_combiner_is_implemented(self):
        # A combiner in the list but not in the dispatch would silently fall
        # through to "last".
        weights = {
            combine: _weight(
                build_weighted_snapshot(REPEATED, WeightSpec(combine=combine)), "a", "b"
            )
            for combine in COMBINERS
        }

        assert len(set(weights.values())) >= 4, f"combiners are not distinct: {weights}"

    def test_a_pair_seen_once_is_unchanged_by_the_combiner(self):
        for combine in COMBINERS:
            snapshot = build_weighted_snapshot(REPEATED, WeightSpec(combine=combine))
            assert _weight(snapshot, "b", "c") == pytest.approx(5.0)

    def test_the_evidence_behind_a_weight_is_reported(self):
        snapshot = build_weighted_snapshot(REPEATED)
        edge = next(e for e in snapshot.edges if (e.src, e.dst) == ("a", "b"))

        assert edge.count == 2
        assert edge.first_timestamp == 100
        assert edge.last_timestamp == 200

    def test_last_takes_the_most_recent_not_the_last_listed(self):
        # Newest first: "last" must still pick the 200-second transaction.
        reversed_order = list(reversed(REPEATED))

        snapshot = build_weighted_snapshot(reversed_order, WeightSpec(combine="last"))

        assert _weight(snapshot, "a", "b") == pytest.approx(30.0)


class TestWeightSources:
    def test_amount_uses_the_transferred_value(self):
        snapshot = build_weighted_snapshot(REPEATED, WeightSpec(source="amount"))

        assert _weight(snapshot, "b", "c") == pytest.approx(5.0)

    def test_count_ignores_amounts_entirely(self):
        snapshot = build_weighted_snapshot(REPEATED, WeightSpec(source="count"))

        assert _weight(snapshot, "a", "b") == pytest.approx(2.0)
        assert _weight(snapshot, "b", "c") == pytest.approx(1.0)

    def test_recency_halves_the_weight_every_half_life(self):
        edges = [
            {"src": "a", "dst": "b", "timestamp": 1000},
            {"src": "c", "dst": "d", "timestamp": 900},
            {"src": "e", "dst": "f", "timestamp": 800},
        ]

        snapshot = build_weighted_snapshot(
            edges, WeightSpec(source="recency", combine="last", half_life_seconds=100)
        )

        assert _weight(snapshot, "a", "b") == pytest.approx(1.0)
        assert _weight(snapshot, "c", "d") == pytest.approx(0.5)
        assert _weight(snapshot, "e", "f") == pytest.approx(0.25)

    def test_recency_uses_the_newest_edge_as_now_by_default(self):
        edges = [{"src": "a", "dst": "b", "timestamp": 500}]

        snapshot = build_weighted_snapshot(edges, WeightSpec(source="recency"))

        # Anchored to the data, not to the wall clock — a rebuilt snapshot
        # must be identical, not decayed further.
        assert _weight(snapshot, "a", "b") == pytest.approx(1.0)

    def test_an_explicit_reference_timestamp_is_honoured(self):
        edges = [{"src": "a", "dst": "b", "timestamp": 500}]

        snapshot = build_weighted_snapshot(
            edges,
            WeightSpec(source="recency", reference_timestamp=600, half_life_seconds=100),
        )

        assert _weight(snapshot, "a", "b") == pytest.approx(0.5)

    def test_an_edge_newer_than_the_reference_is_not_over_weighted(self):
        edges = [{"src": "a", "dst": "b", "timestamp": 900}]

        snapshot = build_weighted_snapshot(
            edges,
            WeightSpec(source="recency", reference_timestamp=500, half_life_seconds=100),
        )

        # Ages clamp at zero; a clock skew must not hand one edge a weight
        # that swamps every other.
        assert _weight(snapshot, "a", "b") == pytest.approx(1.0)

    def test_every_source_is_usable(self):
        for source in WEIGHT_SOURCES:
            snapshot = build_weighted_snapshot(REPEATED, WeightSpec(source=source))
            assert len(snapshot.edges) == 2


class TestNormalisation:
    def test_weights_scale_so_the_largest_is_one(self):
        snapshot = build_weighted_snapshot(REPEATED, WeightSpec(normalize=True))

        assert max(edge.weight for edge in snapshot.edges) == pytest.approx(1.0)
        assert _weight(snapshot, "b", "c") == pytest.approx(5.0 / 40.0)

    def test_normalising_all_zero_weights_does_not_divide_by_zero(self):
        edges = [{"src": "a", "dst": "b", "amount": 0.0, "timestamp": 1}]

        snapshot = build_weighted_snapshot(edges, WeightSpec(normalize=True))

        assert _weight(snapshot, "a", "b") == 0.0


class TestDeterminism:
    def test_edge_order_does_not_change_the_result(self):
        shuffled = list(REPEATED)
        random.Random(4242).shuffle(shuffled)

        assert build_weighted_snapshot(REPEATED) == build_weighted_snapshot(shuffled)

    def test_every_combiner_is_order_independent(self):
        shuffled = list(REPEATED)
        random.Random(7).shuffle(shuffled)

        for combine in COMBINERS:
            spec = WeightSpec(combine=combine)
            assert build_weighted_snapshot(REPEATED, spec) == build_weighted_snapshot(
                shuffled, spec
            )

    def test_a_tie_on_timestamp_breaks_the_same_way_every_run(self):
        # Two transactions at the same instant: "last" has to pick one, and
        # it must pick the same one whatever order they arrive in.
        tied = [
            {"src": "a", "dst": "b", "amount": 1.0, "timestamp": 100},
            {"src": "a", "dst": "b", "amount": 2.0, "timestamp": 100},
        ]

        forwards = build_weighted_snapshot(tied, WeightSpec(combine="last"))
        backwards = build_weighted_snapshot(list(reversed(tied)), WeightSpec(combine="last"))

        assert _weight(forwards, "a", "b") == _weight(backwards, "a", "b")

    def test_nodes_and_edges_come_out_sorted(self):
        snapshot = build_weighted_snapshot(REPEATED)

        assert list(snapshot.nodes) == sorted(snapshot.nodes)
        pairs = [(edge.src, edge.dst) for edge in snapshot.edges]
        assert pairs == sorted(pairs)

    def test_the_spec_travels_with_the_snapshot(self):
        spec = WeightSpec(source="count", combine="max")

        snapshot = build_weighted_snapshot(REPEATED, spec)

        # Two combiners disagree on the same input, so a snapshot is only
        # reproducible if it records which one produced it.
        assert snapshot.spec == spec
        assert snapshot.to_dict()["spec"]["combine"] == "max"


class TestValidation:
    def test_an_edge_without_endpoints_is_rejected(self):
        with pytest.raises(InvalidWeightError, match="missing src or dst"):
            build_weighted_snapshot([{"src": "a"}])

    def test_a_non_numeric_amount_is_rejected(self):
        with pytest.raises(InvalidWeightError, match="not numeric"):
            build_weighted_snapshot([{"src": "a", "dst": "b", "amount": "lots"}])

    def test_a_negative_amount_is_rejected(self):
        with pytest.raises(InvalidWeightError, match="negative"):
            build_weighted_snapshot([{"src": "a", "dst": "b", "amount": -1.0}])

    def test_an_infinite_amount_is_rejected(self):
        with pytest.raises(InvalidWeightError, match="finite"):
            build_weighted_snapshot([{"src": "a", "dst": "b", "amount": float("inf")}])

    def test_weighting_by_amount_without_amounts_is_refused(self):
        # Quietly treating a missing amount as zero would drop the edge's
        # influence with no indication anything was wrong.
        with pytest.raises(InvalidWeightError, match="carries no amount"):
            build_weighted_snapshot([{"src": "a", "dst": "b"}], WeightSpec(source="amount"))

    def test_weighting_by_recency_without_timestamps_is_refused(self):
        with pytest.raises(InvalidWeightError, match="carries no timestamp"):
            build_weighted_snapshot(
                [{"src": "a", "dst": "b", "amount": 1.0}], WeightSpec(source="recency")
            )

    def test_an_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="unknown weight source"):
            build_weighted_snapshot(REPEATED, WeightSpec(source="vibes"))

    def test_an_unknown_combiner_is_rejected(self):
        with pytest.raises(ValueError, match="unknown combiner"):
            build_weighted_snapshot(REPEATED, WeightSpec(combine="median"))

    def test_a_non_positive_half_life_is_rejected(self):
        with pytest.raises(ValueError, match="half_life_seconds"):
            build_weighted_snapshot(REPEATED, WeightSpec(source="recency", half_life_seconds=0))

    def test_an_empty_snapshot_is_valid_and_empty(self):
        snapshot = build_weighted_snapshot([])

        assert snapshot.nodes == ()
        assert snapshot.edges == ()


class TestSnapshotEdgeCompatibility:
    def test_snapshot_edge_dataclasses_are_accepted(self):
        from astroml.features.graph.snapshot import Edge

        snapshot = build_weighted_snapshot([Edge(src="a", dst="b", timestamp=1, amount=7.0)])

        assert _weight(snapshot, "a", "b") == pytest.approx(7.0)

    def test_a_positional_edge_without_an_amount_still_constructs(self):
        from astroml.features.graph.snapshot import Edge

        # The amount field was appended for this issue; three-argument
        # construction must keep working.
        snapshot = build_weighted_snapshot([Edge("a", "b", 1)], WeightSpec(source="count"))

        assert _weight(snapshot, "a", "b") == pytest.approx(1.0)


class TestEdgeIndexOutput:
    def test_edge_index_and_weights_line_up(self):
        snapshot = build_weighted_snapshot(REPEATED)

        edge_index, weights = snapshot.to_edge_index()

        assert len(edge_index) == 2
        assert len(edge_index[0]) == len(edge_index[1]) == len(weights)

    def test_indices_address_the_node_list(self):
        snapshot = build_weighted_snapshot(REPEATED)

        (rows, cols), _ = snapshot.to_edge_index()

        for row, col in zip(rows, cols):
            assert 0 <= row < len(snapshot.nodes)
            assert 0 <= col < len(snapshot.nodes)

    def test_the_pairs_round_trip_back_to_node_names(self):
        snapshot = build_weighted_snapshot(REPEATED)

        (rows, cols), weights = snapshot.to_edge_index()

        recovered = {
            (snapshot.nodes[row], snapshot.nodes[col]): weight
            for row, col, weight in zip(rows, cols, weights)
        }
        assert recovered[("a", "b")] == pytest.approx(40.0)

    def test_an_empty_snapshot_produces_empty_tensors(self):
        edge_index, weights = build_weighted_snapshot([]).to_edge_index()

        assert edge_index == [[], []]
        assert weights == []
