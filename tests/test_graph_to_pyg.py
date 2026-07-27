"""
Tests for graph_to_pyg_data function.
"""

import pytest
import torch
import numpy as np
from astroml.graph_utils import graph_to_pyg_data


class TestGraphToPyG:
    """Test cases for graph_to_pyg_data function."""
    
    def test_basic_conversion(self):
        """Test basic conversion with node features and edge index."""
        node_features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3 nodes, 2 features
        edge_index = [[0, 1, 2], [1, 2, 0]]  # 3 edges, COO format
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Check node features
        assert data.x.shape[0] == 3  # num_nodes
        assert data.x.shape[1] == 2  # num_node_features
        assert data.x.dtype == torch.float32
        
        # Check edge index
        assert data.edge_index.shape[0] == 2  # COO format
        assert data.edge_index.shape[1] == 3  # num_edges
        assert data.edge_index.dtype == torch.int64
        
        # Check no edge attributes
        assert data.edge_attr is None
        
        # Check no labels
        assert data.y is None
    
    def test_conversion_with_edge_features(self):
        """Test conversion with edge features included."""
        node_features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        edge_index = [[0, 1, 2], [1, 2, 0]]
        edge_features = [[0.5], [0.6], [0.7]]  # 3 edges, 1 feature
        
        data = graph_to_pyg_data(node_features, edge_index, edge_features)
        
        # Check edge attributes
        assert data.edge_attr is not None
        assert data.edge_attr.shape[0] == 3  # num_edges
        assert data.edge_attr.shape[1] == 1  # num_edge_features
        assert data.edge_attr.dtype == torch.float32
    
    def test_conversion_with_node_labels(self):
        """Test conversion with node labels."""
        node_features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        edge_index = [[0, 1, 2], [1, 2, 0]]
        node_labels = [0, 1, 0]  # 3 node labels
        
        data = graph_to_pyg_data(node_features, edge_index, node_labels=node_labels)
        
        # Check labels
        assert data.y is not None
        assert data.y.shape[0] == 3  # num_nodes

    def test_conversion_with_numpy_edge_features_and_node_labels(self):
        """Test conversion with numpy arrays for edge features and labels."""
        node_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int32)
        edge_features = np.array([[0.5], [0.6]], dtype=np.float64)
        node_labels = np.array([0, 1], dtype=np.int64)

        data = graph_to_pyg_data(node_features, edge_index, edge_features, node_labels)

        assert data.edge_attr.dtype == torch.float32
        assert data.y.dtype == torch.int64
        assert data.y.shape == (2,)

    def test_invalid_edge_index_negative_id(self):
        """Test error handling for negative edge index values."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0, -1], [1, 0]]

        with pytest.raises(ValueError, match="Edge index contains negative node IDs"):
            graph_to_pyg_data(node_features, edge_index)

    def test_invalid_node_labels_2d_shape(self):
        """Test error handling for node labels with incorrect dimensionality."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0, 1], [1, 0]]
        node_labels = [[0], [1]]

        with pytest.raises(ValueError, match="node_labels must be 1D array"):
            graph_to_pyg_data(node_features, edge_index, node_labels=node_labels)

    def test_edge_index_format_conversion(self):
        """Test edge index format conversion from [num_edges, 2] to [2, num_edges]."""
        node_features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        edge_index = [[0, 1], [1, 2], [2, 0]]  # [num_edges, 2] format
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Check that edge index is in COO format [2, num_edges]
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 3
        
        # Verify the edges are correct
        expected_edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        assert torch.equal(data.edge_index, expected_edges)
    
    def test_numpy_array_inputs(self):
        """Test conversion with numpy array inputs."""
        node_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int32)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        assert data.x.shape[0] == 2
        assert data.x.shape[1] == 2
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 2
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.int64
    
    def test_empty_graph(self):
        """Test empty graph handling."""
        node_features = []  # Empty list
        edge_index = [[], []]  # Empty edges
        
        # Convert to numpy arrays
        node_features = np.array(node_features, dtype=np.float32).reshape(0, 2)
        edge_index = np.array(edge_index, dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        assert data.x.shape[0] == 0
        assert data.edge_index.shape[1] == 0
    
    def test_single_node_no_edges(self):
        """Test single node with no edges."""
        node_features = [[1.0, 2.0]]  # 1 node, 2 features
        edge_index = [[], []]  # No edges
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        assert data.x.shape[0] == 1
        assert data.x.shape[1] == 2
        assert data.edge_index.shape[1] == 0
    
    def test_single_edge(self):
        """Test graph with single edge."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0], [1]]  # Single edge
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        assert data.x.shape[0] == 2
        assert data.edge_index.shape[1] == 1
        assert torch.equal(data.edge_index, torch.tensor([[0], [1]], dtype=torch.int64))
    
    def test_invalid_node_features_shape(self):
        """Test error handling for invalid node features shape."""
        node_features = [1.0, 2.0, 3.0]  # 1D array instead of 2D
        edge_index = [[0, 1], [1, 0]]
        
        with pytest.raises(ValueError, match="node_features must be 2D array"):
            graph_to_pyg_data(node_features, edge_index)
    
    def test_invalid_edge_index_shape(self):
        """Test error handling for invalid edge index shape."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0, 1, 2], [1, 2], [0, 1]]  # Invalid shape
        
        with pytest.raises(ValueError, match="edge_index must"):
            graph_to_pyg_data(node_features, edge_index)
    
    def test_edge_index_out_of_bounds(self):
        """Test error handling for edge index out of bounds."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]  # 2 nodes
        edge_index = [[0, 2], [1, 0]]  # Node ID 2 is out of bounds
        
        with pytest.raises(ValueError, match="Edge index contains node ID"):
            graph_to_pyg_data(node_features, edge_index)
    
    def test_edge_features_shape_mismatch(self):
        """Test error handling for edge features shape mismatch."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0, 1], [1, 0]]  # 2 edges
        edge_features = [[0.5]]  # Only 1 edge feature instead of 2
        
        with pytest.raises(ValueError, match="edge_features shape mismatch"):
            graph_to_pyg_data(node_features, edge_index, edge_features)
    
    def test_node_labels_shape_mismatch(self):
        """Test error handling for node labels shape mismatch."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]  # 2 nodes
        edge_index = [[0, 1], [1, 0]]
        node_labels = [0]  # Only 1 label instead of 2
        
        with pytest.raises(ValueError, match="node_labels shape mismatch"):
            graph_to_pyg_data(node_features, edge_index, node_labels=node_labels)
    
    def test_edge_features_zero_dim(self):
        """Test edge features with zero-dimensional features per edge."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0], [1]]
        edge_features = [[]]  # 1 edge, 0 features

        data = graph_to_pyg_data(node_features, edge_index, edge_features)

        assert data.edge_attr is not None
        assert data.edge_attr.shape == (1, 0)

    def test_ambiguous_2x2_edge_index(self):
        """Test edge_index with shape [2, 2] which is both valid [2, N] and [N, 2]."""
        node_features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        edge_index = [[0, 1], [2, 0]]  # [2, 2] — valid as COO

        data = graph_to_pyg_data(node_features, edge_index)

        assert data.edge_index.shape == (2, 2)
        expected = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
        assert torch.equal(data.edge_index, expected)

    def test_node_features_int_dtype(self):
        """Test node_features with integer dtype converts to float32."""
        node_features = np.array([[1, 2], [3, 4]], dtype=np.int32)
        edge_index = [[0], [1]]

        data = graph_to_pyg_data(node_features, edge_index)

        assert data.x.dtype == torch.float32
        assert torch.equal(data.x, torch.tensor([[1., 2.], [3., 4.]]))

    def test_node_labels_numpy_int(self):
        """Test node_labels as numpy int array."""
        node_features = [[1.0, 2.0], [3.0, 4.0]]
        edge_index = [[0], [1]]
        node_labels = np.array([0, 1], dtype=np.int32)

        data = graph_to_pyg_data(node_features, edge_index, node_labels=node_labels)

        assert data.y is not None
        assert data.y.dtype == torch.int64
        assert torch.equal(data.y, torch.tensor([0, 1], dtype=torch.int64))

    def test_complete_graph_example(self):
        """Test with a complete graph example including all features."""
        # 4 nodes, 3 features each
        node_features = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]
        
        # 6 edges in a complete graph (4 choose 2)
        edge_index = [
            [0, 0, 0, 1, 1, 2],  # source nodes
            [1, 2, 3, 2, 3, 3]   # target nodes
        ]
        
        # Edge features (2 features per edge)
        edge_features = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
            [1.1, 1.2]
        ]
        
        # Node labels
        node_labels = [0, 1, 0, 1]
        
        data = graph_to_pyg_data(node_features, edge_index, edge_features, node_labels)
        
        # Verify all components
        assert data.x.shape == (4, 3)
        assert data.edge_index.shape == (2, 6)
        assert data.edge_attr.shape == (6, 2)
        assert data.y.shape == (4,)
        assert data.num_nodes == 4
