"""Self-supervised link prediction model for AstroML.

Predicts whether two accounts will transact in the next N ledgers.

Architecture
------------
* **Encoder** — a stack of GCN layers (reuses :class:`~astroml.models.gcn.GCN`
  internals) that produces one embedding vector per node.
* **Decoder** — scores a candidate edge (u, v) using either a dot product
  between the two node embeddings or a small MLP over their concatenation.

The model is trained with a binary cross-entropy objective on positive
(observed future) edges and randomly sampled negative (non-)edges — a
standard self-supervised link prediction setup.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """GCN encoder that produces node embeddings.

    Args:
        input_dim: Dimension of input node features.
        hidden_dims: Sizes of intermediate GCN layers.
        embedding_dim: Size of the final node embedding.
        dropout: Dropout probability applied between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        dims = [input_dim] + hidden_dims + [embedding_dim]
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings of shape ``[N, embedding_dim]``."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """Link prediction model for self-supervised training.

    Combines a GCN encoder with a scoring decoder to predict whether
    two accounts will transact within the next N ledgers.

    Args:
        input_dim: Dimension of input node features.
        hidden_dims: Sizes of intermediate GCN encoder layers.
        embedding_dim: Size of the final node embedding.
        dropout: Dropout applied inside the encoder.
        decoder: ``"dot"`` uses a dot product; ``"mlp"`` uses a two-layer
            MLP over the concatenated pair embeddings.

    Example::

        model = LinkPredictor(input_dim=128, hidden_dims=[64], embedding_dim=32)
        z = model.encode(x, edge_index)           # [N, 32]
        scores = model.decode(z, pos_edge_index)  # [E] logits
        loss = model.loss(z, pos_edge_index, neg_edge_index)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        dropout: float = 0.5,
        decoder: Literal["dot", "mlp"] = "dot",
    ) -> None:
        super().__init__()
        self.decoder_type = decoder
        self.encoder = GCNEncoder(input_dim, hidden_dims, embedding_dim, dropout)

        if decoder == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the GCN encoder and return node embeddings ``[N, embedding_dim]``."""
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score candidate edges.

        Args:
            z: Node embeddings ``[N, embedding_dim]``.
            edge_index: Candidate edges ``[2, E]``.

        Returns:
            Raw logits ``[E]`` (apply sigmoid for probabilities).
        """
        src, dst = edge_index[0], edge_index[1]
        if self.decoder_type == "dot":
            return (z[src] * z[dst]).sum(dim=-1)
        else:
            pair = torch.cat([z[src], z[dst]], dim=-1)
            return self.mlp(pair).squeeze(-1)

    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        """Score every possible node pair (dense, O(N²)).

        Only suitable for small graphs / evaluation.  Returns a ``[N, N]``
        matrix of raw logits.
        """
        if self.decoder_type == "dot":
            return z @ z.t()
        else:
            N = z.size(0)
            src = torch.arange(N, device=z.device).repeat_interleave(N)
            dst = torch.arange(N, device=z.device).repeat(N)
            pair = torch.cat([z[src], z[dst]], dim=-1)
            return self.mlp(pair).squeeze(-1).view(N, N)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss over positive and negative edges.

        Args:
            z: Node embeddings produced by :meth:`encode`.
            pos_edge_index: Positive (observed future) edges ``[2, E_pos]``.
            neg_edge_index: Negative (sampled non-)edges ``[2, E_neg]``.

        Returns:
            Scalar BCE loss.
        """
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_scores.size(0), device=z.device),
                torch.zeros(neg_scores.size(0), device=z.device),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    # ------------------------------------------------------------------
    # Convenience forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        query_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode then decode query edges.

        Args:
            x: Node features ``[N, input_dim]``.
            edge_index: Graph connectivity used for message passing ``[2, E]``.
            query_edge_index: Edges to score ``[2, Q]``.

        Returns:
            Raw logits ``[Q]``.
        """
        z = self.encode(x, edge_index)
        return self.decode(z, query_edge_index)
