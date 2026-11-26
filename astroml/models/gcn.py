from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Standard 2-layer Graph Convolutional Network for node classification.

    Architecture: GCNConv -> ReLU -> Dropout -> GCNConv -> log_softmax

    Isolated nodes are valid input. Each node gets a self-loop, so a node with no
    neighbours is represented by its own features alone, and a graph with no edges
    at all works. Use :func:`astroml.models.graph_input.prepare_graph_input` to
    check tensors before a run (non-finite features, out-of-range indices).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index) -> Any:
        if edge_index.numel() == 0 and edge_index.dtype != torch.long:
            # ``torch.tensor([[], []])`` is float32, which the scatter kernel
            # rejects. An empty edge list has no indices to get wrong.
            edge_index = edge_index.to(torch.long)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
