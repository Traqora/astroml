import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Configurable Graph Convolutional Network for node classification.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))

        # Output layer
        self.convs.append(GCNConv(hidden_dims[-1], output_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)