from __future__ import annotations

import torch

from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.training.train_sage import build_reconstruction_target, train_epoch


def _make_synthetic_graph(num_nodes=20, num_edges=40):
    """Create a random graph with features."""
    edge_index = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )
    features = torch.randn(num_nodes, 8)
    return edge_index, features


def test_train_epoch_loss_decreases():
    """Training for a few epochs should decrease loss."""
    torch.manual_seed(42)
    edge_index, features = _make_synthetic_graph(20, 40)

    encoder = InductiveSAGEEncoder(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        aggregator="mean",
    )
    sampler = MultiHopSampler(edge_index, num_nodes=20, fanout=[5, 3])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    train_nodes = torch.arange(20)

    loss1 = train_epoch(
        encoder, sampler, features, edge_index, train_nodes, optimizer, batch_size=10
    )
    for _ in range(9):
        loss_n = train_epoch(
            encoder, sampler, features, edge_index, train_nodes, optimizer, batch_size=10
        )

    assert loss_n < loss1


def test_build_reconstruction_target():
    """Reconstruction target is mean of neighbor raw features."""
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
    features = torch.tensor(
        [
            [1.0, 0.0],  # node 0
            [0.0, 1.0],  # node 1
            [1.0, 1.0],  # node 2
            [0.0, 0.0],  # node 3 (target)
        ]
    )
    target_nodes = torch.tensor([3])

    targets = build_reconstruction_target(edge_index, features, target_nodes)

    expected = torch.tensor([[2.0 / 3.0, 2.0 / 3.0]])
    assert torch.allclose(targets, expected, atol=1e-5)
