from __future__ import annotations

import os
import sys
import types

import pytest

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def test_gat_multihead_shapes_and_attention_sum():
    import torch
    from astroml.features.gnn.attention import GATConv

    # Simple 3-node graph with edges: 0->1, 2->1, 1->2
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)
    x = torch.randn(3, 5)

    layer = GATConv(in_dim=5, out_dim=4, heads=3, concat=True, dropout=0.0)
    out, attn = layer(x, edge_index, return_attention=True)

    # Output shape: [N, H * F]
    assert out.shape == (3, 3 * 4)

    # Attention shape: [E, H]
    assert attn.shape == (edge_index.size(1), 3)

    # Verify attention sums to 1 over incoming edges for each head at each dst node present in edges
    dst = edge_index[1]
    for v in dst.unique():
        mask = (dst == v)
        a = attn[mask]  # [E_v, H]
        colsum = a.sum(dim=0)
        assert torch.allclose(colsum, torch.ones_like(colsum), atol=1e-5)


def test_gat_export_attention():
    import torch
    from astroml.features.gnn.attention import GATConv

    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)
    x = torch.randn(3, 4)

    layer = GATConv(in_dim=4, out_dim=3, heads=2)
    _ = layer(x, edge_index)  # not returning attention

    exported = layer.export_attention()
    assert exported is not None
    eidx, attn = exported
    assert eidx.shape == edge_index.shape
    assert attn.shape == (edge_index.size(1), 2)
