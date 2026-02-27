from __future__ import annotations

import pytest
import torch
from astroml.features.gnn.sage import SAGEConv

def test_sage_conv_shapes():
    # Simple 3-node graph: 0->1, 2->1
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    x = torch.randn(3, 8)
    
    conv = SAGEConv(in_dim=8, out_dim=4, aggregator='mean')
    out = conv(x, edge_index)
    
    # Num nodes is 3
    assert out.shape == (3, 4)

def test_sage_conv_bipartite_shapes():
    # Bipartite graph or sampled subgraph
    # src nodes (0, 1, 2), dst nodes (0, 1)
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.long)
    x_src = torch.randn(3, 8)
    x_dst = torch.randn(2, 8)
    
    conv = SAGEConv(in_dim=8, out_dim=4, aggregator='mean')
    out = conv((x_src, x_dst), edge_index)
    
    # Output nodes match dst nodes
    assert out.shape == (2, 4)

def test_sage_aggregator_mean():
    # 0->1, 2->1
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    x = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])
    
    conv = SAGEConv(in_dim=2, out_dim=2, aggregator='mean', bias=False)
    # Identity weights for testing
    with torch.no_grad():
        conv.lin_l.weight.copy_(torch.eye(2))
        conv.lin_r.weight.copy_(torch.eye(2))
        
    out = conv(x, edge_index)
    
    # Node 1 gets mean of node 0 and 2: (1+3)/2 = 2
    # Update for node 1: lin_l(mean_aggr) + lin_r(x_1) = [2,2] + [2,2] = [4,4]
    assert torch.allclose(out[1], torch.tensor([4.0, 4.0]))
    # Node 0 has no incoming edges, aggr is 0
    # Update for node 0: [0,0] + [1,1] = [1,1]
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0]))

def test_sample_neighbors():
    from astroml.features.gnn.sage import sample_neighbors
    # 0->1, 2->1, 1->0
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 0]], dtype=torch.long)
    nodes = torch.tensor([1])
    
    src, dst = sample_neighbors(edge_index, nodes, num_samples=1)
    
    assert src.numel() == 1
    assert dst.numel() == 1
    assert dst[0] == 1
    assert src[0] in [0, 2]
