from __future__ import annotations

"""Inductive GraphSAGE training with unsupervised reconstruction loss.

The encoder learns to produce embeddings that can reconstruct the mean
of each node's immediate neighbors' raw features. This produces general-
purpose embeddings without requiring labels.
"""

import logging

import torch
from torch import nn

from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder

logger = logging.getLogger(__name__)


def build_reconstruction_target(
    edge_index: torch.Tensor,
    features: torch.Tensor,
    target_nodes: torch.Tensor,
) -> torch.Tensor:
    """Compute mean neighbor features for each target node.

    Parameters
    ----------
    edge_index : Tensor [2, E]
        Full graph edge index.
    features : Tensor [N, F]
        Raw node features.
    target_nodes : Tensor [B]
        Nodes to compute targets for.

    Returns
    -------
    Tensor [B, F] - mean of neighbors' features per target node.
    """
    src, dst = edge_index
    result = torch.zeros(target_nodes.size(0), features.size(1))

    for i, node in enumerate(target_nodes):
        neighbors = src[dst == node]
        if neighbors.numel() > 0:
            result[i] = features[neighbors].mean(dim=0)

    return result


def train_epoch(
    encoder: InductiveSAGEEncoder,
    sampler: MultiHopSampler,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    train_nodes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 512,
    device: str = "cpu",
) -> float:
    """Train encoder for one epoch using mini-batch reconstruction loss.

    Parameters
    ----------
    encoder : InductiveSAGEEncoder
    sampler : MultiHopSampler
    features : Tensor [N, F] - raw node features
    edge_index : Tensor [2, E] - full graph edges (for reconstruction targets)
    train_nodes : Tensor [T] - node indices to train on
    optimizer : torch.optim.Optimizer
    batch_size : int
    device : str

    Returns
    -------
    float - average loss for the epoch.
    """
    encoder.train()
    decoder = nn.Linear(encoder.convs[-1].out_dim, features.size(1)).to(device)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=optimizer.defaults["lr"])

    perm = torch.randperm(train_nodes.size(0))
    total_loss = 0.0
    num_batches = 0

    for start in range(0, train_nodes.size(0), batch_size):
        batch_indices = perm[start: start + batch_size]
        batch_nodes = train_nodes[batch_indices]

        # Sample neighborhoods
        adjs, sampled_node_ids = sampler.sample(batch_nodes)

        # Slice features for sampled nodes
        x = features[sampled_node_ids].to(device)

        # Forward — sampler returns adjs innermost-first; encoder expects outermost-first.
        # After encoding, slice to the number of target nodes (sampled_node_ids places
        # target nodes at indices [0..len(batch_nodes)-1]).
        embeddings = encoder(x, list(reversed(adjs)))
        embeddings = embeddings[: batch_nodes.size(0)]

        # Reconstruction target: mean of neighbors' raw features
        targets = build_reconstruction_target(edge_index, features, batch_nodes).to(device)

        # Decode and compute loss
        decoded = decoder(embeddings)
        loss = nn.functional.mse_loss(decoded, targets)

        optimizer.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        optimizer.step()
        decoder_opt.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
