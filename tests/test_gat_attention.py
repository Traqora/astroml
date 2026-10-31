from __future__ import annotations

import pytest

try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def test_gat_multihead_shapes_and_attention_sum():
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
        mask = dst == v
        a = attn[mask]  # [E_v, H]
        colsum = a.sum(dim=0)
        assert torch.allclose(colsum, torch.ones_like(colsum), atol=1e-5)


def test_gat_export_attention():
    from astroml.features.gnn.attention import GATConv  # noqa: E402

    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)
    x = torch.randn(3, 4)

    layer = GATConv(in_dim=4, out_dim=3, heads=2)
    _ = layer(x, edge_index)  # not returning attention

    exported = layer.export_attention()
    assert exported is not None
    eidx, attn = exported
    assert eidx.shape == edge_index.shape
    assert attn.shape == (edge_index.size(1), 2)


def test_gat_dropout_is_reproducible_under_a_fixed_seed():
    """Same seed + training mode + nonzero dropout -> identical output on
    every run (GitHub Issue #842). Neither existing test above exercises
    dropout > 0 at all; without this, a change that broke determinism
    (e.g. swapping nn.Dropout for a module that doesn't respect the global
    RNG) would go unnoticed."""
    from astroml.features.gnn.attention import GATConv

    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)

    def run_once():
        torch.manual_seed(42)
        x = torch.randn(3, 5)
        layer = GATConv(in_dim=5, out_dim=4, heads=3, concat=True, dropout=0.5)
        layer.train()
        torch.manual_seed(123)  # separately seed the dropout mask draw
        return layer(x, edge_index)

    out1 = run_once()
    out2 = run_once()
    assert torch.equal(out1, out2), (
        "GATConv with dropout > 0 produced different output across two runs "
        "seeded identically, dropout masking is not deterministic"
    )


def test_gat_dropout_disabled_in_eval_mode():
    """In eval() mode, dropout must be a no-op regardless of seed: calling
    forward twice without reseeding must still produce identical output,
    since there is no dropout mask being drawn (GitHub Issue #842)."""
    from astroml.features.gnn.attention import GATConv

    torch.manual_seed(7)
    x = torch.randn(3, 5)
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)

    layer = GATConv(in_dim=5, out_dim=4, heads=3, concat=True, dropout=0.5)
    layer.eval()

    with torch.no_grad():
        out1 = layer(x, edge_index)
        out2 = layer(x, edge_index)  # no reseed between calls

    assert torch.equal(out1, out2), (
        "GATConv produced different output across two eval()-mode calls "
        "with no reseeding, dropout should be disabled in eval mode"
    )
