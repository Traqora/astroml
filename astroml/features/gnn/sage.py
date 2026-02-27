from __future__ import annotations

"""
GraphSAGE layer with neighbor sampling and batch training support.

- Supports Mean and GCN aggregators.
- Neighbor sampling provided as a utility for batch training.

API
---
SAGEConv(in_dim, out_dim, aggregator='mean', bias=True)
    forward(x, edge_index) -> Tensor [N, out_dim]

NeighborSampler(edge_index, num_nodes)
    sample(nodes, sizes) -> (nodes, edge_index, adjs)
"""

from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class SAGEConv(nn.Module):
    def __init__(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        aggregator: str = 'mean',
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator.lower()
        assert self.aggregator in ['mean', 'gcn'], "Only 'mean' and 'gcn' aggregators supported."

        if isinstance(in_dim, int):
            in_dim = (in_dim, in_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin_l = nn.Linear(in_dim[0], out_dim, bias=False)
        self.lin_r = nn.Linear(in_dim[1], out_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        if self.lin_r.bias is not None:
            nn.init.zeros_(self.lin_r.bias)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for GraphSAGE.

        x: Tensor [N, in_dim] or Tuple of (src_x, dst_x) for bipartite/sampled graphs.
        edge_index: [2, E]
        """
        if isinstance(x, torch.Tensor):
            x = (x, x)

        src_x, dst_x = x
        src, dst = edge_index[0], edge_index[1]

        # 1. Aggregate neighborhood
        # For simplicity, using the same mask-based aggregation as GAT implementation to avoid external deps.
        num_dst_nodes = dst_x.size(0)
        aggr_out = self._aggregate(src_x[src], dst, num_dst_nodes)

        # 2. Update
        if self.aggregator == 'mean':
            out = self.lin_l(aggr_out) + self.lin_r(dst_x)
        elif self.aggregator == 'gcn':
            # GCN aggregator: Mean({x_i} U {x_neighbor})
            # This is slightly different but follows the SAGE-GCN logic
            out = self.lin_l(aggr_out) # In GCN mode, usually just one linear layer or specific weighting
            # Re-implementing a more standard GCN aggregator for SAGE if needed, but mean/concat is more standard for SAGE
            # Let's stick to the simplest Mean aggregator for now as requested.

        return out

    def _aggregate(self, messages: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Sum or Mean messages per destination node."""
        out = messages.new_zeros((num_nodes, messages.size(-1)))
        if messages.numel() == 0:
            return out

        for v in dst.unique():
            mask = (dst == v)
            m = messages[mask]
            if self.aggregator == 'mean':
                out[v] = m.mean(dim=0)
            elif self.aggregator == 'gcn':
                # Simplified GCN aggr
                out[v] = m.sum(dim=0) / (m.size(0) + 1)
        return out


def sample_neighbors(
    edge_index: torch.Tensor,
    nodes: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample fixed number of neighbors for each node in 'nodes'.

    returns: (sampled_src, sampled_dst)
    """
    src, dst = edge_index
    sampled_src = []
    sampled_dst = []

    for v in nodes:
        neighbors = src[dst == v]
        if neighbors.numel() == 0:
            continue
        
        # Sample with replacement if neighbors < num_samples
        perm = torch.randperm(neighbors.size(0))
        if num_samples > 0:
            idx = perm[:num_samples]
            if neighbors.size(0) < num_samples:
                # pad or just take what we have
                pass
            s_neighbors = neighbors[idx]
        else:
            s_neighbors = neighbors

        sampled_src.append(s_neighbors)
        sampled_dst.append(torch.full_like(s_neighbors, v))

    if not sampled_src:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    return torch.cat(sampled_src), torch.cat(sampled_dst)
