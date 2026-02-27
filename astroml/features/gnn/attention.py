from __future__ import annotations

"""
Graph Attention Network (GAT) layer with multi-head attention and exportable attention weights.

- Multi-head attention supported via heads parameter
- Attention weights can be returned from forward(return_attention=True)
  and are also stored on the module (layer.last_attention_) for export/inspection.

API
---
GATConv(in_dim, out_dim, heads=4, concat=True, dropout=0.0, negative_slope=0.2)
  forward(x, edge_index, return_attention=False) -> output or (output, attn)

Inputs
------
- x: Tensor [N, in_dim]
- edge_index: LongTensor [2, E], where rows are [src, dst] for each edge

Outputs
-------
- output: Tensor [N, heads*out_dim] if concat else [N, out_dim]
- attn (optional): Tensor [E, heads] attention weights per edge and head

Notes
-----
- This implementation avoids external deps (PyG, torch-scatter) for portability.
  Softmax normalization over incoming edges is computed using per-node masking. This
  is efficient enough for small/medium graphs and unit tests. Replace the
  segment operations with scatter-based reductions for large graphs when needed.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute softmax per destination node for multi-head scores.

    scores: [E, H]
    dst: [E]
    returns: [E, H] where for each head h and node v: sum_{e: dst_e=v} softmax_h(e) = 1
    """
    attn = torch.zeros_like(scores)
    if scores.numel() == 0:
        return attn
    # Iterate distinct destination nodes present in edges
    for v in dst.unique():
        mask = dst == v
        s = scores[mask]  # [E_v, H]
        # Numerically stable softmax per head
        s_max = s.max(dim=0, keepdim=True).values
        exp_s = (s - s_max).exp()
        denom = exp_s.sum(dim=0, keepdim=True).clamp_min(1e-9)
        attn[mask] = exp_s / denom
    return attn


def _aggregate_sum_by_dst(messages: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sum messages per destination node using simple masking aggregation.

    messages: [E, H, F]
    dst: [E]
    returns: [N, H, F]
    """
    N, H, Fdim = num_nodes, messages.shape[1], messages.shape[2]
    out = messages.new_zeros((N, H, Fdim))
    if messages.numel() == 0:
        return out
    for v in dst.unique():
        mask = dst == v
        out[v] = messages[mask].sum(dim=0)
    return out


class GATConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # Attention vector per head over concatenated [Wh_i || Wh_j]
        self.att = nn.Parameter(torch.empty(heads, 2 * out_dim))
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim * heads if concat else out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # Store last attention for export: tuple(edge_index, attn [E, H])
        self.last_attention_: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAT output and optionally return attention weights.

        x: [N, in_dim]
        edge_index: [2, E] where rows are [src, dst]
        return_attention: if True, also return attn weights [E, heads]
        """
        N = x.size(0)
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be [2, E]"
        src, dst = edge_index[0], edge_index[1]
        E = edge_index.size(1)

        Wh = self.lin(x)  # [N, H*F]
        Wh = Wh.view(N, self.heads, self.out_dim)  # [N, H, F]

        # Prepare per-edge head features
        Wh_i = Wh[src]  # [E, H, F]
        Wh_j = Wh[dst]  # [E, H, F]

        # Compute attention logits per edge per head
        # [E, H, 2F] * [H, 2F] -> [E, H]
        cat_ij = torch.cat([Wh_i, Wh_j], dim=-1)  # [E, H, 2F]
        att_logits = F.leaky_relu((cat_ij * self.att).sum(dim=-1), negative_slope=self.negative_slope)  # [E, H]

        # Normalize over incoming edges per node for each head
        alpha = _segment_softmax(att_logits, dst, N)  # [E, H]
        alpha = self.attn_dropout(alpha)

        # Message passing: weighted sum of neighbor features
        messages = alpha.unsqueeze(-1) * Wh_i  # [E, H, F]
        out = _aggregate_sum_by_dst(messages, dst, N)  # [N, H, F]

        if self.concat:
            out = out.reshape(N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)  # [N, F]

        if self.bias is not None:
            out = out + self.bias

        # Save last attention for export
        self.last_attention_ = (edge_index.detach().clone(), alpha.detach().clone())

        if return_attention:
            return out, alpha
        return out

    def export_attention(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the last computed (edge_index, attention) or None if not computed yet.

        - edge_index: [2, E] LongTensor
        - attention: [E, heads] FloatTensor
        """
        return self.last_attention_
