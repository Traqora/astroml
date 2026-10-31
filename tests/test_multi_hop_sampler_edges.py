"""Edge-case and invariant tests for MultiHopSampler (#898)."""

from __future__ import annotations

import pytest
import torch

from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder


def _random_graph(num_nodes: int, num_edges: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,), generator=gen),
            torch.randint(0, num_nodes, (num_edges,), generator=gen),
        ]
    )


def _assert_layer_invariants(adjs, all_nodes, num_targets):
    """Check the contract InductiveSAGEEncoder relies on."""
    assert adjs, "expected at least one layer"
    # Innermost layer's destinations are exactly the targets.
    assert adjs[-1][1][1] == num_targets
    prev_num_dst = None
    for edge_index, (num_src, num_dst) in adjs:
        assert num_src <= all_nodes.numel()
        assert num_dst <= num_src
        if edge_index.numel():
            assert int(edge_index[0].max()) < num_src
            assert int(edge_index[1].max()) < num_dst
        if prev_num_dst is not None:
            # Rows produced by the previous layer are the sources of this one.
            assert num_src == prev_num_dst
        prev_num_dst = num_dst
    # Outermost layer covers every sampled node.
    assert adjs[0][1][0] == all_nodes.numel()


def test_empty_fanout_returns_targets_only():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index, num_nodes=3, fanout=[])
    adjs, all_nodes = sampler.sample(torch.tensor([2, 0]))
    assert adjs == []
    assert all_nodes.tolist() == [2, 0]


def test_empty_edge_index():
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    sampler = MultiHopSampler(edge_index, num_nodes=4, fanout=[3, 2])
    adjs, all_nodes = sampler.sample(torch.tensor([1, 3]))
    assert len(adjs) == 2
    for edge, (num_src, num_dst) in adjs:
        assert edge.shape == (2, 0)
        assert (num_src, num_dst) == (2, 2)
    assert all_nodes.tolist() == [1, 3]


def test_targets_come_first_in_order():
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 4, 5, 5]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index, num_nodes=6, fanout=[2])
    adjs, all_nodes = sampler.sample(torch.tensor([5, 4]))
    assert all_nodes[:2].tolist() == [5, 4]
    assert set(all_nodes.tolist()) == {0, 1, 2, 3, 4, 5}


def test_local_edges_map_back_to_real_edges():
    edge_index = _random_graph(15, 40, seed=1)
    real = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    sampler = MultiHopSampler(edge_index, num_nodes=15, fanout=[4, 3])
    torch.manual_seed(0)
    adjs, all_nodes = sampler.sample(torch.tensor([0, 1, 2]))
    ids = all_nodes.tolist()
    for edge, _ in adjs:
        for s, d in zip(edge[0].tolist(), edge[1].tolist()):
            assert (ids[s], ids[d]) in real


def test_self_loops_and_cycles():
    # 0 -> 0 self-loop and a 0 -> 1 -> 2 -> 0 cycle
    edge_index = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index, num_nodes=3, fanout=[2, 2, 2])
    torch.manual_seed(0)
    adjs, all_nodes = sampler.sample(torch.tensor([0]))
    assert len(adjs) == 3
    assert sorted(all_nodes.tolist()) == sorted(set(all_nodes.tolist()))
    _assert_layer_invariants(adjs, all_nodes, num_targets=1)


def test_duplicate_targets_are_collapsed():
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index, num_nodes=3, fanout=[2])
    adjs, all_nodes = sampler.sample(torch.tensor([2, 2]))
    assert all_nodes.tolist().count(2) == 1
    assert all_nodes[0].item() == 2


@pytest.mark.parametrize("seed", range(25))
def test_layer_invariants_hold_on_random_graphs(seed):
    edge_index = _random_graph(20, 50, seed=seed)
    sampler = MultiHopSampler(edge_index, num_nodes=20, fanout=[5, 3])
    torch.manual_seed(seed)
    targets = torch.tensor([0, 3, 7])
    adjs, all_nodes = sampler.sample(targets)
    _assert_layer_invariants(adjs, all_nodes, num_targets=3)


@pytest.mark.parametrize("seed", range(25))
def test_encoder_accepts_sampled_layers(seed):
    """Regression: two-hop samples used to index past the feature matrix."""
    edge_index = _random_graph(12, 30, seed=seed)
    sampler = MultiHopSampler(edge_index, num_nodes=12, fanout=[3, 2])
    encoder = InductiveSAGEEncoder(
        input_dim=4, hidden_dim=8, output_dim=4, num_layers=2, dropout=0.0
    )
    torch.manual_seed(seed)
    targets = torch.tensor([0, 1])
    adjs, all_nodes = sampler.sample(targets)
    x = torch.randn(all_nodes.numel(), 4)
    out = encoder(x, adjs)
    assert out.shape == (2, 4)
