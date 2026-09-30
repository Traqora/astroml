from __future__ import annotations

"""End-to-end test: train GraphSAGE on subgraph A, produce embeddings for
unseen nodes in subgraph B. Verifies the core inductive learning property."""

import torch

from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.training.train_sage import train_epoch


def test_inductive_generalization():
    """Embeddings for unseen nodes are non-zero and vary across nodes."""
    torch.manual_seed(123)

    # --- Subgraph A (training): nodes 0-9, dense ---
    num_train_nodes = 10
    train_edges = []
    for i in range(num_train_nodes):
        for j in range(num_train_nodes):
            if i != j and torch.rand(1).item() > 0.5:
                train_edges.append([i, j])
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    train_features = torch.randn(num_train_nodes, 8)

    # --- Train encoder ---
    encoder = InductiveSAGEEncoder(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        aggregator="mean",
    )
    sampler = MultiHopSampler(train_edge_index, num_train_nodes, fanout=[5, 3])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    train_nodes = torch.arange(num_train_nodes)

    for _ in range(20):
        train_epoch(
            encoder, sampler, train_features, train_edge_index, train_nodes, optimizer, batch_size=5
        )

    # --- Subgraph B (unseen): nodes 0-4, completely disjoint accounts ---
    num_test_nodes = 5
    test_edges = []
    for i in range(num_test_nodes):
        for j in range(num_test_nodes):
            if i != j and torch.rand(1).item() > 0.5:
                test_edges.append([i, j])
    if not test_edges:
        test_edges = [[0, 1], [1, 0]]  # ensure at least one edge
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    test_features = torch.randn(num_test_nodes, 8)

    # --- Embed unseen nodes using the trained encoder ---
    test_sampler = MultiHopSampler(test_edge_index, num_test_nodes, fanout=[5, 3])
    target_nodes = torch.arange(num_test_nodes)

    encoder.train(False)
    with torch.no_grad():
        adjs, sampled = test_sampler.sample(target_nodes)
        x = test_features[sampled]
        # Sampler returns adjs innermost-first; encoder expects outermost-first.
        embeddings = encoder(x, list(reversed(adjs)))

    # Verify: embeddings are non-zero
    assert embeddings.shape == (num_test_nodes, 8)
    assert (embeddings.abs().sum(dim=1) > 0).all(), "All embeddings should be non-zero"

    # Verify: embeddings vary across nodes (not all identical)
    pairwise_diffs = torch.cdist(embeddings, embeddings)
    off_diagonal = pairwise_diffs[~torch.eye(num_test_nodes, dtype=torch.bool)]
    assert (off_diagonal > 1e-6).any(), "Embeddings should vary across nodes"

    # Verify: embeddings are finite
    assert torch.isfinite(embeddings).all()
