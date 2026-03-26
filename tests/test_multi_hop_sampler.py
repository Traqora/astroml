from __future__ import annotations

import torch
from astroml.features.gnn.sampler import MultiHopSampler


def test_single_hop_sampling():
    """Single-hop with fanout [2] returns one adjacency layer."""
    edge_index = torch.tensor([[0, 1, 3, 0], [2, 2, 2, 1]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=4, fanout=[2])

    target = torch.tensor([2], dtype=torch.long)
    adjs, all_nodes = sampler.sample(target)

    assert len(adjs) == 1
    edge_idx, size = adjs[0]
    assert edge_idx.shape[0] == 2
    assert edge_idx.shape[1] <= 2
    assert 2 in all_nodes.tolist()


def test_two_hop_sampling():
    """Two-hop sampling returns two adjacency layers."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=4, fanout=[2, 2])

    target = torch.tensor([3], dtype=torch.long)
    adjs, all_nodes = sampler.sample(target)

    assert len(adjs) == 2
    assert all_nodes[0].item() == 3
    node_list = all_nodes.tolist()
    assert 2 in node_list


def test_isolated_node():
    """Isolated target node returns empty adjacency lists."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=3, fanout=[5])

    target = torch.tensor([2], dtype=torch.long)
    adjs, all_nodes = sampler.sample(target)

    assert len(adjs) == 1
    edge_idx, size = adjs[0]
    assert edge_idx.shape[1] == 0
    assert 2 in all_nodes.tolist()


def test_deterministic_with_seed():
    """Same seed produces same sampling results."""
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=6, fanout=[2])
    target = torch.tensor([5], dtype=torch.long)

    torch.manual_seed(42)
    adjs1, nodes1 = sampler.sample(target)

    torch.manual_seed(42)
    adjs2, nodes2 = sampler.sample(target)

    assert torch.equal(adjs1[0][0], adjs2[0][0])
    assert torch.equal(nodes1, nodes2)
