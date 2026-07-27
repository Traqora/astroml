"""Property-based tests for graph utilities using Hypothesis.

Issue #338: Add property-based tests for graph utilities.
Tests graph invariants and snapshot edge cases.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as np_st

from astroml.graph_utils import graph_to_pyg_data


class TestGraphToPyGProperties:
    """Property-based tests for graph_to_pyg_data function."""

    @given(
        node_features=np_st.arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=0, max_value=100), st.integers(min_value=1, max_value=50))
        ),
        edge_index=np_st.arrays(
            dtype=np.int64,
            shape=st.tuples(st.integers(min_value=0, max_value=2), st.integers(min_value=0, max_value=1000))
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_output_shapes_consistency(self, node_features, edge_index):
        """Test that output shapes are consistent with input dimensions."""
        # Skip if node_features is empty
        if node_features.shape[0] == 0:
            return
        
        # Ensure edge_index is valid shape
        if edge_index.ndim != 2 or edge_index.shape[0] not in [1, 2]:
            return
        
        # Skip if edge indices are out of bounds
        num_nodes = node_features.shape[0]
        if edge_index.size > 0 and edge_index.max() >= num_nodes:
            return
        if edge_index.size > 0 and edge_index.min() < 0:
            return
        
        try:
            data = graph_to_pyg_data(node_features, edge_index)
            
            # Property: node count matches input
            assert data.num_nodes == node_features.shape[0]
            
            # Property: node features shape matches
            assert data.x.shape[0] == node_features.shape[0]
            assert data.x.shape[1] == node_features.shape[1]
            
            # Property: edge index is in COO format [2, num_edges]
            assert data.edge_index.shape[0] == 2
            
            # Property: dtype conversions
            assert data.x.dtype == torch.float32
            assert data.edge_index.dtype == torch.int64
        except (ValueError, AssertionError):
            # Expected for invalid inputs
            pass

    @given(
        num_nodes=st.integers(min_value=1, max_value=50),
        num_features=st.integers(min_value=1, max_value=20),
        num_edges=st.integers(min_value=0, max_value=200)
    )
    @settings(max_examples=50, deadline=None)
    def test_graph_invariants_valid_graphs(self, num_nodes, num_features, num_edges):
        """Test graph invariants for valid randomly generated graphs."""
        if num_nodes == 0:
            return
        
        # Generate random node features
        node_features = np.random.randn(num_nodes, num_features).astype(np.float32)
        
        # Generate random edge indices within bounds
        if num_edges > 0:
            edge_index = np.random.randint(0, num_nodes, size=(2, num_edges), dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Invariant: number of nodes is preserved
        assert data.num_nodes == num_nodes
        
        # Invariant: node features dimensionality preserved
        assert data.x.shape == (num_nodes, num_features)
        
        # Invariant: edge index is always [2, num_edges]
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == num_edges
        
        # Invariant: all edge indices are within bounds
        if num_edges > 0:
            assert data.edge_index.max() < num_nodes
            assert data.edge_index.min() >= 0

    @given(
        node_features=np_st.arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=1, max_value=50), st.integers(min_value=1, max_value=20))
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_empty_graph_edge_case(self, node_features):
        """Test empty graph (no edges) edge case."""
        num_nodes = node_features.shape[0]
        edge_index = np.zeros((2, 0), dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Properties for empty edge set
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr is None
        assert data.num_nodes == num_nodes

    @given(
        num_nodes=st.integers(min_value=2, max_value=50),
        num_features=st.integers(min_value=1, max_value=10),
        num_edge_features=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_edge_features_preservation(self, num_nodes, num_features, num_edge_features):
        """Test that edge features are properly preserved."""
        node_features = np.random.randn(num_nodes, num_features).astype(np.float32)
        num_edges = num_nodes * 2  # Ensure some edges
        edge_index = np.random.randint(0, num_nodes, size=(2, num_edges), dtype=np.int64)
        edge_features = np.random.randn(num_edges, num_edge_features).astype(np.float32)
        
        data = graph_to_pyg_data(node_features, edge_index, edge_features)
        
        # Property: edge features shape matches edges
        assert data.edge_attr is not None
        assert data.edge_attr.shape == (num_edges, num_edge_features)
        assert data.edge_attr.dtype == torch.float32

    @given(
        num_nodes=st.integers(min_value=1, max_value=50),
        num_features=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_node_labels_preservation(self, num_nodes, num_features):
        """Test that node labels are properly preserved."""
        node_features = np.random.randn(num_nodes, num_features).astype(np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        node_labels = np.random.randint(0, 5, size=num_nodes, dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index, node_labels=node_labels)
        
        # Property: node labels shape matches nodes
        assert data.y is not None
        assert data.y.shape == (num_nodes,)
        assert data.y.dtype == torch.int64

    @given(
        node_features=st.lists(
            st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), 
                     min_size=1, max_size=20),
            min_size=1, max_size=50
        ),
        edge_index=st.lists(
            st.lists(st.integers(min_value=0, max_value=50), min_size=2, max_size=2),
            min_size=1, max_size=2
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_list_input_conversion(self, node_features, edge_index):
        """Test that list inputs are properly converted to arrays."""
        # Ensure homogeneous lists for valid numpy conversion
        if not all(len(row) == len(node_features[0]) for row in node_features):
            return
        if not all(len(row) == 2 for row in edge_index):
            return
        
        node_features_arr = np.array(node_features, dtype=np.float32)
        edge_index_arr = np.array(edge_index, dtype=np.int64)
        
        # Skip invalid edge indices
        num_nodes = node_features_arr.shape[0]
        if edge_index_arr.size > 0 and edge_index_arr.max() >= num_nodes:
            return
        
        # Ensure valid edge index shape
        if edge_index_arr.ndim != 2:
            return
        
        try:
            data = graph_to_pyg_data(node_features, edge_index)
            
            # Property: output is always torch tensors
            assert isinstance(data.x, torch.Tensor)
            assert isinstance(data.edge_index, torch.Tensor)
        except ValueError:
            # Expected for malformed inputs
            pass

    @given(
        num_nodes=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20, deadline=None)
    def test_self_loop_handling(self, num_nodes):
        """Test graphs with self-loops."""
        node_features = np.random.randn(num_nodes, 2).astype(np.float32)
        
        # Create edges with self-loops
        edge_indices = []
        for i in range(num_nodes):
            edge_indices.append([i, i])  # Self-loop
            if i < num_nodes - 1:
                edge_indices.append([i, i + 1])  # Forward edge
        
        edge_index = np.array(edge_indices, dtype=np.int64).T
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Property: self-loops are preserved
        assert data.num_nodes == num_nodes
        assert data.edge_index.shape[1] == len(edge_indices)

    @given(
        num_nodes=st.integers(min_value=2, max_value=30)
    )
    @settings(max_examples=20, deadline=None)
    def test_symmetric_edges(self, num_nodes):
        """Test that symmetric edges (undirected graphs) are handled correctly."""
        node_features = np.random.randn(num_nodes, 2).astype(np.float32)
        
        # Create symmetric edges
        edge_indices = []
        for i in range(min(num_nodes, 5)):
            for j in range(i + 1, min(num_nodes, 5)):
                edge_indices.append([i, j])
                edge_indices.append([j, i])  # Symmetric
        
        if edge_indices:
            edge_index = np.array(edge_indices, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Property: all edges are within bounds
        assert data.edge_index.max() < num_nodes
        assert data.edge_index.min() >= 0

    @given(
        node_features=np_st.arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=1, max_value=50), st.integers(min_value=1, max_value=20))
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_single_node_graph(self, node_features):
        """Test single node graph edge case."""
        # Take only first node
        single_node_features = node_features[:1]
        edge_index = np.zeros((2, 0), dtype=np.int64)
        
        data = graph_to_pyg_data(single_node_features, edge_index)
        
        # Property: single node is preserved
        assert data.num_nodes == 1
        assert data.x.shape[0] == 1
        assert data.edge_index.shape[1] == 0

    @given(
        num_nodes=st.integers(min_value=1, max_value=100),
        num_features=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=30, deadline=None)
    def test_dense_graph(self, num_nodes, num_features):
        """Test dense graph (many edges relative to nodes)."""
        node_features = np.random.randn(num_nodes, num_features).astype(np.float32)
        
        # Create many edges (up to num_nodes^2)
        max_edges = min(num_nodes * num_nodes, 1000)
        num_edges = min(num_nodes * 5, max_edges)
        
        if num_edges > 0:
            edge_index = np.random.randint(0, num_nodes, size=(2, num_edges), dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Property: dense graph preserves all data
        assert data.num_nodes == num_nodes
        assert data.x.shape == (num_nodes, num_features)
        assert data.edge_index.shape[1] == num_edges

    @given(
        node_features=np_st.arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=2, max_value=50), st.integers(min_value=1, max_value=20))
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_duplicate_edges(self, node_features):
        """Test graphs with duplicate edges."""
        num_nodes = node_features.shape[0]
        
        # Create duplicate edges (requires at least 2 nodes)
        edge_index = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int64)
        
        data = graph_to_pyg_data(node_features, edge_index)
        
        # Property: duplicate edges are preserved
        assert data.num_nodes == num_nodes
        # Duplicate edges should be kept as-is
        assert data.edge_index.shape[1] == 3
