from __future__ import annotations

import torch

from astroml.models.sage_encoder import InductiveSAGEEncoder


def test_encoder_output_shape():
    """Encoder produces correct output dimensions."""
    encoder = InductiveSAGEEncoder(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        aggregator="mean",
    )
    x = torch.randn(10, 8)
    adj0_edge = torch.tensor([[7, 8, 9, 6], [0, 1, 2, 3]], dtype=torch.long)
    adj0 = (adj0_edge, (10, 6))
    adj1_edge = torch.tensor([[3, 4, 5], [0, 1, 2]], dtype=torch.long)
    adj1 = (adj1_edge, (6, 3))

    adjs = [adj0, adj1]
    out = encoder(x, adjs)

    assert out.shape == (3, 8)


def test_encoder_single_layer():
    """Single-layer encoder works correctly."""
    encoder = InductiveSAGEEncoder(
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_layers=1,
        dropout=0.0,
        aggregator="mean",
    )
    x = torch.randn(5, 4)
    adj_edge = torch.tensor([[3, 4], [0, 1]], dtype=torch.long)
    adjs = [(adj_edge, (5, 2))]

    out = encoder(x, adjs)
    assert out.shape == (2, 4)


def test_encoder_gradient_flow():
    """Gradients flow through all layers."""
    encoder = InductiveSAGEEncoder(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        num_layers=2,
        dropout=0.0,
        aggregator="mean",
    )
    x = torch.randn(6, 4, requires_grad=True)
    adj0_edge = torch.tensor([[3, 4, 5], [0, 1, 2]], dtype=torch.long)
    adj1_edge = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    adjs = [(adj0_edge, (6, 3)), (adj1_edge, (3, 1))]

    out = encoder(x, adjs)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    for conv in encoder.convs:
        assert conv.lin_l.weight.grad is not None
        assert conv.lin_r.weight.grad is not None
