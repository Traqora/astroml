"""Neighbourhood feature precomputation (issue #735).

The three properties the issue asks for are what these tests pin: the output
is deterministic whatever order the edges arrive in, a malformed edge is
rejected at the boundary rather than turned into a NaN, and the cache is
keyed such that a stale vector can never be served.
"""

from __future__ import annotations

import json
import random

import pytest

from astroml.features.graph.neighborhood import (
    FEATURE_VERSION,
    InvalidGraphError,
    NeighborhoodConfig,
    NeighborhoodFeatureCache,
    NeighborhoodFeatures,
    graph_fingerprint,
    normalize_edges,
    precompute_neighborhood_features,
)

# a -> b -> c -> d, plus a shortcut a -> c and a back edge d -> a.
CHAIN = [
    {"src": "a", "dst": "b", "amount": 10.0},
    {"src": "b", "dst": "c", "amount": 20.0},
    {"src": "c", "dst": "d", "amount": 30.0},
    {"src": "a", "dst": "c", "amount": 5.0},
    {"src": "d", "dst": "a", "amount": 1.0},
]


class TestDeterminism:
    def test_edge_order_does_not_change_the_result(self):
        shuffled = list(CHAIN)
        random.Random(1234).shuffle(shuffled)

        assert precompute_neighborhood_features(CHAIN) == precompute_neighborhood_features(shuffled)

    def test_nodes_are_emitted_in_sorted_order(self):
        features = precompute_neighborhood_features(CHAIN)

        assert features.nodes == ("a", "b", "c", "d")
        assert list(features.nodes) == sorted(features.nodes)

    def test_the_matrix_column_order_is_fixed(self):
        first = precompute_neighborhood_features(CHAIN)
        second = precompute_neighborhood_features(list(reversed(CHAIN)))

        assert first.feature_names == second.feature_names
        assert first.to_matrix() == second.to_matrix()

    def test_the_graph_fingerprint_ignores_edge_order(self):
        shuffled = list(CHAIN)
        random.Random(7).shuffle(shuffled)

        assert graph_fingerprint(normalize_edges(CHAIN)) == graph_fingerprint(
            normalize_edges(shuffled)
        )

    def test_a_changed_amount_changes_the_fingerprint(self):
        altered = [dict(edge) for edge in CHAIN]
        altered[0]["amount"] = 10.5

        assert graph_fingerprint(normalize_edges(CHAIN)) != graph_fingerprint(
            normalize_edges(altered)
        )


class TestDegreeAndAmountFeatures:
    def test_degrees_are_counted_per_direction(self):
        row = precompute_neighborhood_features(CHAIN).for_node("a")

        # a sends to b and c; d sends to a.
        assert row["out_degree"] == 2
        assert row["in_degree"] == 1
        assert row["degree"] == 3

    def test_amounts_are_summed_and_averaged(self):
        row = precompute_neighborhood_features(CHAIN).for_node("a")

        assert row["total_sent"] == pytest.approx(15.0)
        assert row["mean_sent_amount"] == pytest.approx(7.5)
        assert row["max_sent_amount"] == pytest.approx(10.0)
        assert row["total_received"] == pytest.approx(1.0)

    def test_a_node_that_only_receives_has_zero_send_features(self):
        features = precompute_neighborhood_features([{"src": "x", "dst": "y", "amount": 3.0}])
        row = features.for_node("y")

        assert row["out_degree"] == 0
        assert row["total_sent"] == 0.0
        # Zero, not NaN — a division by an empty neighbourhood must not reach
        # the feature matrix.
        assert row["mean_sent_amount"] == 0.0

    def test_parallel_edges_count_separately_from_unique_neighbours(self):
        features = precompute_neighborhood_features(
            [
                {"src": "a", "dst": "b", "amount": 1.0},
                {"src": "a", "dst": "b", "amount": 2.0},
            ]
        )
        row = features.for_node("a")

        assert row["out_degree"] == 2
        assert row["unique_out_neighbors"] == 1

    def test_a_missing_amount_is_treated_as_zero(self):
        features = precompute_neighborhood_features([{"src": "a", "dst": "b"}])

        assert features.for_node("a")["total_sent"] == 0.0


class TestHopFeatures:
    def test_hop_one_is_the_immediate_neighbourhood(self):
        features = precompute_neighborhood_features(CHAIN)

        # a -> {b, c}
        assert features.for_node("a")["hop1_size"] == 2

    def test_hop_two_adds_the_next_ring(self):
        features = precompute_neighborhood_features(CHAIN)

        # a -> {b, c} -> {d}
        assert features.for_node("a")["hop2_size"] == 3

    def test_hop_counts_are_non_decreasing(self):
        features = precompute_neighborhood_features(CHAIN, NeighborhoodConfig(max_hops=4))

        for node in features.nodes:
            row = features.for_node(node)
            sizes = [row[f"hop{k}_size"] for k in range(1, 5)]
            assert sizes == sorted(sizes), f"{node} hop sizes went backwards: {sizes}"

    def test_a_node_never_counts_itself(self):
        features = precompute_neighborhood_features(
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "a"}],
            NeighborhoodConfig(max_hops=3),
        )

        # The cycle returns to 'a', which must not inflate its own count.
        assert features.for_node("a")["hop3_size"] == 1

    def test_undirected_expansion_reaches_further_than_directed(self):
        edges = [{"src": "a", "dst": "b"}, {"src": "c", "dst": "b"}]

        directed = precompute_neighborhood_features(edges, NeighborhoodConfig(directed=True))
        undirected = precompute_neighborhood_features(edges, NeighborhoodConfig(directed=False))

        # a -> b is a dead end when following direction; ignoring direction,
        # c is reachable through b.
        assert directed.for_node("a")["hop2_size"] == 1
        assert undirected.for_node("a")["hop2_size"] == 2

    def test_the_hop_budget_bounds_the_reachable_count(self):
        # A star: the hub reaches 20 nodes in one hop.
        edges = [{"src": "hub", "dst": f"leaf{i:02d}"} for i in range(20)]

        bounded = precompute_neighborhood_features(
            edges, NeighborhoodConfig(max_hops=2, hop_node_budget=5)
        )

        # Truncated, never overstated.
        assert bounded.for_node("hub")["hop1_size"] <= 20


class TestNeighbourAggregates:
    def test_neighbour_mean_degree_averages_over_the_neighbours(self):
        edges = [
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ]
        features = precompute_neighborhood_features(edges)

        # a's only neighbour is b, whose degree is 1 in + 2 out = 3.
        assert features.for_node("a")["neighbour_mean_degree"] == pytest.approx(3.0)

    def test_a_node_with_no_neighbours_aggregates_to_zero(self):
        features = precompute_neighborhood_features([{"src": "a", "dst": "b"}])

        assert features.for_node("b")["neighbour_mean_degree"] == 0.0
        assert features.for_node("b")["neighbour_mean_amount"] == 0.0

    def test_aggregates_can_be_switched_off(self):
        features = precompute_neighborhood_features(
            CHAIN, NeighborhoodConfig(neighbour_aggregates=False)
        )

        assert "neighbour_mean_degree" not in features.feature_names


class TestValidation:
    def test_an_edge_without_a_destination_is_rejected(self):
        with pytest.raises(InvalidGraphError, match="missing src or dst"):
            precompute_neighborhood_features([{"src": "a"}])

    def test_an_empty_endpoint_is_rejected(self):
        with pytest.raises(InvalidGraphError, match="empty endpoint"):
            precompute_neighborhood_features([{"src": "a", "dst": ""}])

    def test_a_non_numeric_amount_is_rejected(self):
        with pytest.raises(InvalidGraphError, match="not numeric"):
            precompute_neighborhood_features([{"src": "a", "dst": "b", "amount": "lots"}])

    def test_a_nan_amount_is_rejected(self):
        with pytest.raises(InvalidGraphError, match="finite"):
            precompute_neighborhood_features([{"src": "a", "dst": "b", "amount": float("nan")}])

    def test_a_negative_amount_is_rejected(self):
        with pytest.raises(InvalidGraphError, match="negative"):
            precompute_neighborhood_features([{"src": "a", "dst": "b", "amount": -1.0}])

    def test_max_hops_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="max_hops"):
            precompute_neighborhood_features(CHAIN, NeighborhoodConfig(max_hops=0))

    def test_an_empty_graph_is_valid_and_empty(self):
        features = precompute_neighborhood_features([])

        assert features.nodes == ()
        assert features.to_matrix() == ((), [])

    def test_an_unknown_node_raises_a_clear_error(self):
        features = precompute_neighborhood_features(CHAIN)

        with pytest.raises(KeyError, match="not in this graph"):
            features.for_node("zzz")


class TestSnapshotEdgeCompatibility:
    def test_snapshot_edge_dataclasses_are_accepted(self):
        from astroml.features.graph.snapshot import Edge

        features = precompute_neighborhood_features(
            [Edge(src="a", dst="b", timestamp=1, amount=4.0)]
        )

        assert features.for_node("a")["total_sent"] == pytest.approx(4.0)

    def test_a_positional_edge_without_an_amount_still_works(self):
        from astroml.features.graph.snapshot import Edge

        # The amount field was added for the reporting work; existing
        # three-argument construction must keep working.
        features = precompute_neighborhood_features([Edge("a", "b", 1)])

        assert features.for_node("a")["out_degree"] == 1


class TestCaching:
    def test_a_second_call_is_served_from_cache(self):
        cache = NeighborhoodFeatureCache()

        first = precompute_neighborhood_features(CHAIN, cache=cache)
        second = precompute_neighborhood_features(CHAIN, cache=cache)

        assert second is first, "the identical graph must not be recomputed"

    def test_a_changed_graph_is_a_miss(self):
        cache = NeighborhoodFeatureCache()
        extended = CHAIN + [{"src": "d", "dst": "e", "amount": 2.0}]

        first = precompute_neighborhood_features(CHAIN, cache=cache)
        second = precompute_neighborhood_features(extended, cache=cache)

        assert second is not first
        assert "e" in second.nodes

    def test_a_changed_config_is_a_miss(self):
        cache = NeighborhoodFeatureCache()

        two = precompute_neighborhood_features(CHAIN, NeighborhoodConfig(max_hops=2), cache=cache)
        three = precompute_neighborhood_features(CHAIN, NeighborhoodConfig(max_hops=3), cache=cache)

        assert "hop3_size" not in two.feature_names
        assert "hop3_size" in three.feature_names

    def test_the_cache_key_carries_the_feature_version(self):
        key = NeighborhoodFeatureCache.make_key("cfg", "graph")

        assert FEATURE_VERSION in key

    def test_a_disk_cache_survives_a_new_process(self, tmp_path):
        first = precompute_neighborhood_features(CHAIN, cache=NeighborhoodFeatureCache(tmp_path))
        # A fresh cache object with no warm memory — the hit has to come off
        # disk or not at all.
        second = precompute_neighborhood_features(CHAIN, cache=NeighborhoodFeatureCache(tmp_path))

        assert second == first
        assert list(tmp_path.glob("*.json"))

    def test_an_entry_from_an_older_feature_version_is_not_served(self, tmp_path):
        cache = NeighborhoodFeatureCache(tmp_path)
        features = precompute_neighborhood_features(CHAIN, cache=cache)
        key = NeighborhoodFeatureCache.make_key(
            features.config_fingerprint, features.graph_fingerprint
        )

        stale = features.to_dict() | {"version": "0.0.1"}
        (tmp_path / f"{key}.json").write_text(json.dumps(stale), encoding="utf-8")

        assert NeighborhoodFeatureCache(tmp_path).get(key) is None

    def test_a_corrupt_cache_file_is_a_miss_not_a_crash(self, tmp_path):
        cache = NeighborhoodFeatureCache(tmp_path)
        features = precompute_neighborhood_features(CHAIN, cache=cache)
        key = NeighborhoodFeatureCache.make_key(
            features.config_fingerprint, features.graph_fingerprint
        )
        (tmp_path / f"{key}.json").write_text("{ truncated", encoding="utf-8")

        # A bad cache file must cost a recomputation, never a failed run.
        assert NeighborhoodFeatureCache(tmp_path).get(key) is None

    def test_clear_removes_disk_entries(self, tmp_path):
        cache = NeighborhoodFeatureCache(tmp_path)
        precompute_neighborhood_features(CHAIN, cache=cache)

        cache.clear()

        assert not list(tmp_path.glob("nbhd-v*.json"))


class TestSerialisation:
    def test_a_round_trip_through_json_preserves_everything(self):
        features = precompute_neighborhood_features(CHAIN)

        restored = NeighborhoodFeatures.from_dict(json.loads(json.dumps(features.to_dict())))

        assert restored == features
