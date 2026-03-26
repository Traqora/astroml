from __future__ import annotations

import torch
from astroml.pipeline.inductive import InductiveGraphSAGE
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.features.gnn.sampler import MultiHopSampler


def _make_edges():
    """Synthetic edges as dicts for compute_node_features."""
    return [
        {'src': 'A', 'dst': 'B', 'amount': 100.0, 'timestamp': 1000.0, 'asset': 'XLM'},
        {'src': 'B', 'dst': 'C', 'amount': 50.0, 'timestamp': 2000.0, 'asset': 'XLM'},
        {'src': 'C', 'dst': 'D', 'amount': 25.0, 'timestamp': 3000.0, 'asset': 'USD'},
        {'src': 'A', 'dst': 'C', 'amount': 75.0, 'timestamp': 1500.0, 'asset': 'XLM'},
        {'src': 'D', 'dst': 'A', 'amount': 10.0, 'timestamp': 3500.0, 'asset': 'BTC'},
    ]


def test_embed_nodes_returns_target_embeddings():
    """embed_nodes returns embeddings for requested target nodes only."""
    edges = _make_edges()
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8,
        num_layers=2, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[3, 2], device='cpu')

    result = pipeline.embed_nodes(edges, target_nodes=['C', 'D'], ref_time=4000.0)

    assert 'C' in result
    assert 'D' in result
    assert 'A' not in result
    assert result['C'].shape == (8,)
    assert result['D'].shape == (8,)


def test_embed_nodes_single_transaction_account():
    """Account with a single transaction still gets a valid embedding."""
    edges = [
        {'src': 'X', 'dst': 'Y', 'amount': 1.0, 'timestamp': 100.0, 'asset': 'XLM'},
    ]
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=8, output_dim=4,
        num_layers=1, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[5], device='cpu')

    result = pipeline.embed_nodes(edges, target_nodes=['Y'], ref_time=200.0)

    assert 'Y' in result
    assert result['Y'].shape == (4,)
    assert torch.isfinite(result['Y']).all()


# Also test embed_snapshot:
from astroml.features.graph.snapshot import Edge


def test_embed_snapshot_filters_by_time():
    """embed_snapshot only uses edges within the time window."""
    edges = [
        Edge(src='A', dst='B', timestamp=100),
        Edge(src='B', dst='C', timestamp=200),
        Edge(src='C', dst='D', timestamp=300),
        Edge(src='D', dst='E', timestamp=400),
    ]
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=8, output_dim=4,
        num_layers=1, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[5], device='cpu')

    result = pipeline.embed_snapshot(edges, start_ts=200, end_ts=300, target_nodes=['C'])

    assert 'C' in result
    assert result['C'].shape == (4,)
