"""Connected subgraph sampling for mini-batch training (issue #734).

The properties that matter: every sample is connected (a batch of isolated
nodes defeats the point), the node budget is honoured, the same seed
reproduces the same batch, and the bias knobs actually change what gets
sampled.
"""

from __future__ import annotations

import pytest

from astroml.features.graph.subgraph import (
    SEED_STRATEGIES,
    STRATEGIES,
    InvalidSubgraphRequestError,
    SubgraphSampler,
    sample_connected_subgraph,
)


def _ring(size: int = 30) -> list[dict]:
    """A cycle, so every node has degree 2 and nothing is a hub."""
    return [{"src": f"n{i:02d}", "dst": f"n{(i + 1) % size:02d}"} for i in range(size)]


def _ring_with_hub(size: int = 30, spokes: int = 3) -> list[dict]:
    edges = _ring(size)
    edges += [{"src": "hub", "dst": f"n{i:02d}"} for i in range(0, size, spokes)]
    return edges


def _two_islands() -> list[dict]:
    return [
        {"src": "a1", "dst": "a2"},
        {"src": "a2", "dst": "a3"},
        {"src": "b1", "dst": "b2"},
        {"src": "b2", "dst": "b3"},
    ]


@pytest.fixture()
def sampler() -> SubgraphSampler:
    return SubgraphSampler(_ring_with_hub())


class TestConnectivity:
    def _is_connected(self, subgraph, directed: bool = False) -> bool:
        if len(subgraph) <= 1:
            return True
        adjacency: dict[str, set[str]] = {node: set() for node in subgraph.nodes}
        for src, dst in subgraph.edges:
            adjacency[src].add(dst)
            if not directed:
                adjacency[dst].add(src)

        seen = {subgraph.nodes[0]}
        stack = [subgraph.nodes[0]]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == len(subgraph.nodes)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_every_sample_is_connected(self, sampler, strategy):
        for seed in range(8):
            subgraph = sampler.sample(10, strategy=strategy, seed=seed)
            assert self._is_connected(subgraph), f"{strategy} seed {seed} was disconnected"

    def test_a_sample_never_crosses_into_another_component(self):
        sampler = SubgraphSampler(_two_islands())

        subgraph = sampler.sample(10, seed_node="a1")

        assert set(subgraph.nodes) <= {"a1", "a2", "a3"}

    def test_a_component_smaller_than_the_budget_comes_back_whole(self):
        sampler = SubgraphSampler(_two_islands())

        subgraph = sampler.sample(10, seed_node="b2")

        assert set(subgraph.nodes) == {"b1", "b2", "b3"}
        assert not subgraph.truncated

    def test_an_isolated_pair_is_a_valid_sample(self):
        sampler = SubgraphSampler([{"src": "x", "dst": "y"}])

        subgraph = sampler.sample(5, seed_node="x")

        assert set(subgraph.nodes) == {"x", "y"}


class TestBudget:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("budget", [1, 2, 5, 12])
    def test_the_node_budget_is_never_exceeded(self, sampler, strategy, budget):
        subgraph = sampler.sample(budget, strategy=strategy, seed=3)

        assert len(subgraph) <= budget

    def test_a_budget_of_one_returns_just_the_seed(self, sampler):
        subgraph = sampler.sample(1, seed_node="hub")

        assert subgraph.nodes == ("hub",)

    def test_truncation_is_reported(self, sampler):
        subgraph = sampler.sample(5, seed_node="hub")

        # The ring is far bigger than 5, so the sample is a bounded view.
        assert subgraph.truncated

    def test_a_zero_budget_is_rejected(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="max_nodes"):
            sampler.sample(0)

    def test_a_random_walk_terminates_on_a_small_component(self):
        sampler = SubgraphSampler(_two_islands())

        # Without a step budget the walk would spin forever having already
        # collected the whole component.
        subgraph = sampler.sample(50, seed_node="a1", strategy="random_walk", seed=1)

        assert set(subgraph.nodes) == {"a1", "a2", "a3"}


class TestDeterminism:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_the_same_seed_reproduces_the_sample(self, sampler, strategy):
        first = sampler.sample(10, strategy=strategy, seed=99)
        second = sampler.sample(10, strategy=strategy, seed=99)

        assert first == second

    def test_a_different_seed_gives_a_different_sample(self, sampler):
        samples = {sampler.sample(8, seed=seed).nodes for seed in range(10)}

        assert len(samples) > 1

    def test_a_fresh_sampler_reproduces_the_same_sample(self):
        first = SubgraphSampler(_ring_with_hub()).sample(10, seed=5)
        second = SubgraphSampler(_ring_with_hub()).sample(10, seed=5)

        assert first == second

    def test_edge_order_does_not_change_the_sample(self):
        edges = _ring_with_hub()
        forwards = SubgraphSampler(edges).sample(10, seed=5)
        backwards = SubgraphSampler(list(reversed(edges))).sample(10, seed=5)

        assert forwards == backwards

    def test_nodes_and_edges_come_out_sorted(self, sampler):
        subgraph = sampler.sample(10, seed=1)

        assert list(subgraph.nodes) == sorted(subgraph.nodes)
        assert list(subgraph.edges) == sorted(subgraph.edges)


class TestStrategies:
    def test_a_random_walk_reaches_further_than_bfs(self):
        # A long path: BFS fills the ball around the seed, the walk runs down
        # the line, so the walk's furthest node is further away.
        edges = [{"src": f"p{i:03d}", "dst": f"p{i + 1:03d}"} for i in range(100)]
        sampler = SubgraphSampler(edges)

        bfs = sampler.sample(10, seed_node="p050", strategy="bfs", seed=1)
        walk = sampler.sample(
            10, seed_node="p050", strategy="random_walk", seed=1, restart_probability=0.0
        )

        def spread(subgraph):
            positions = [int(node[1:]) for node in subgraph.nodes]
            return max(positions) - min(positions)

        assert spread(walk) >= spread(bfs)

    def test_bfs_takes_the_immediate_neighbourhood_first(self):
        sampler = SubgraphSampler(_ring())

        subgraph = sampler.sample(3, seed_node="n00", strategy="bfs")

        # The seed's own neighbours, not nodes two hops out.
        assert set(subgraph.nodes) == {"n00", "n01", "n29"}

    def test_an_unknown_strategy_is_rejected(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="unknown strategy"):
            sampler.sample(5, strategy="teleport")

    def test_every_listed_strategy_works(self, sampler):
        for strategy in STRATEGIES:
            assert len(sampler.sample(6, strategy=strategy, seed=1)) >= 1

    def test_restart_probability_must_be_a_probability(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="restart_probability"):
            sampler.sample(5, strategy="random_walk", restart_probability=1.0)


class TestBiasControl:
    def test_degree_seeding_favours_the_hub(self):
        sampler = SubgraphSampler(_ring_with_hub(size=30, spokes=1))

        uniform = [sampler.sample(3, seed=s, seed_strategy="uniform").seed_node for s in range(40)]
        degree = [sampler.sample(3, seed=s, seed_strategy="degree").seed_node for s in range(40)]

        assert degree.count("hub") > uniform.count("hub")

    def test_degree_bias_favours_high_degree_neighbours(self):
        # n00's neighbours are n01, n29 and the hub. With a budget of two,
        # exactly one of them is kept — so how often it is the hub measures
        # the bias directly.
        sampler = SubgraphSampler(_ring_with_hub(size=30, spokes=1))

        def hub_picks(degree_bias: float) -> int:
            return sum(
                "hub" in sampler.sample(2, seed_node="n00", seed=s, degree_bias=degree_bias).nodes
                for s in range(60)
            )

        assert hub_picks(4.0) > hub_picks(0.0)

    def test_bfs_truncation_is_not_alphabetical(self):
        # Regression: adjacency is stored sorted, so walking a frontier ring
        # unshuffled made a truncated sample always keep the
        # alphabetically-first neighbours — a real bias wearing the costume
        # of an arbitrary cut.
        sampler = SubgraphSampler([{"src": "seed", "dst": f"n{i:02d}"} for i in range(10)])

        kept = {sampler.sample(2, seed_node="seed", seed=s).nodes[0] for s in range(40)}

        assert len(kept) > 1, f"truncation always kept the same neighbour: {kept}"

    def test_a_negative_degree_bias_is_rejected(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="degree_bias"):
            sampler.sample(5, degree_bias=-1.0)

    def test_an_unknown_seed_strategy_is_rejected(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="unknown seed_strategy"):
            sampler.sample(5, seed_strategy="clever")

    def test_every_listed_seed_strategy_works(self, sampler):
        for seed_strategy in SEED_STRATEGIES:
            assert sampler.sample(5, seed=1, seed_strategy=seed_strategy).seed_node


class TestDirectedness:
    def test_undirected_by_default_so_a_walk_can_go_upstream(self):
        sampler = SubgraphSampler([{"src": "a", "dst": "b"}])

        assert set(sampler.sample(5, seed_node="b").nodes) == {"a", "b"}

    def test_directed_sampling_only_follows_edges_forwards(self):
        sampler = SubgraphSampler([{"src": "a", "dst": "b"}], directed=True)

        assert sampler.sample(5, seed_node="b").nodes == ("b",)
        assert set(sampler.sample(5, seed_node="a").nodes) == {"a", "b"}


class TestMinibatches:
    def test_it_returns_the_requested_number_of_batches(self, sampler):
        batches = sampler.minibatches(6, num_batches=5, seed=1)

        assert len(batches) == 5

    def test_every_batch_respects_the_budget(self, sampler):
        for batch in sampler.minibatches(6, num_batches=5, seed=1):
            assert len(batch) <= 6

    def test_distinct_seeds_spread_the_batches_around(self, sampler):
        batches = sampler.minibatches(4, num_batches=8, seed=1, distinct_seeds=True)

        # Without this an epoch can quietly become eight passes over one
        # dense neighbourhood.
        seeds = [batch.seed_node for batch in batches]
        assert len(set(seeds)) == len(seeds)

    def test_more_batches_than_nodes_still_succeeds(self):
        sampler = SubgraphSampler(_two_islands())

        batches = sampler.minibatches(2, num_batches=10, seed=1)

        assert len(batches) == 10

    def test_an_epoch_is_reproducible_from_one_seed(self, sampler):
        first = sampler.minibatches(6, num_batches=4, seed=42)
        second = sampler.minibatches(6, num_batches=4, seed=42)

        assert first == second

    def test_a_zero_batch_request_is_rejected(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="num_batches"):
            sampler.minibatches(5, num_batches=0)


class TestValidation:
    def test_an_unknown_seed_node_is_rejected(self, sampler):
        with pytest.raises(InvalidSubgraphRequestError, match="not in this graph"):
            sampler.sample(5, seed_node="nonexistent")

    def test_sampling_an_empty_graph_is_rejected(self):
        with pytest.raises(InvalidSubgraphRequestError, match="empty graph"):
            SubgraphSampler([]).sample(5)

    def test_an_edge_without_endpoints_is_rejected(self):
        with pytest.raises(InvalidSubgraphRequestError, match="missing src or dst"):
            SubgraphSampler([{"src": "a"}])


class TestOutput:
    def test_edge_index_uses_local_indices(self, sampler):
        subgraph = sampler.sample(8, seed=1)

        rows, cols = subgraph.to_edge_index()

        assert len(rows) == len(cols) == subgraph.num_edges
        for row, col in zip(rows, cols):
            assert 0 <= row < len(subgraph)
            assert 0 <= col < len(subgraph)

    def test_every_edge_inside_the_sample_is_included(self):
        sampler = SubgraphSampler(_ring(6))

        subgraph = sampler.sample(6, seed_node="n00")

        # The whole ring fits, so all six edges must survive.
        assert subgraph.num_edges == 6

    def test_the_convenience_wrapper_matches_the_sampler(self):
        edges = _ring_with_hub()

        direct = SubgraphSampler(edges).sample(8, seed=11)
        wrapped = sample_connected_subgraph(edges, 8, seed=11)

        assert direct == wrapped

    def test_the_seed_node_is_always_in_the_sample(self, sampler):
        for strategy in STRATEGIES:
            subgraph = sampler.sample(5, seed_node="hub", strategy=strategy, seed=1)
            assert "hub" in subgraph.nodes


class TestSnapshotEdgeCompatibility:
    def test_snapshot_edge_dataclasses_are_accepted(self):
        from astroml.features.graph.snapshot import Edge

        sampler = SubgraphSampler([Edge("a", "b", 1), Edge("b", "c", 2)])

        assert set(sampler.sample(3, seed_node="a").nodes) == {"a", "b", "c"}


class TestReuse:
    def test_the_adjacency_is_built_once_and_shared(self, sampler):
        # Cheap proxy for "the sampler is reusable": many batches off one
        # instance, no rebuild, all valid.
        batches = [sampler.sample(5, seed=seed) for seed in range(50)]

        assert len(batches) == 50
        assert all(len(batch) <= 5 for batch in batches)

    def test_sampling_does_not_mutate_the_sampler(self, sampler):
        before = sampler.nodes

        sampler.sample(10, seed=1)

        assert sampler.nodes == before
