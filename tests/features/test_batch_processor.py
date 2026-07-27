"""Tests for graph batch processing module."""

from __future__ import annotations

import pytest
from typing import Dict, List

from astroml.features.graph.batch_processor import (
    batch_edges,
    build_adjacency_matrix_sparse,
    get_recommended_batch_size,
    BatchGraphProcessor,
    compute_degree_centrality_batched,
    DEFAULT_BATCH_SIZE,
    BATCH_SIZE_RECOMMENDATIONS,
)


def _make_edges(n: int, num_accounts: int = 100) -> List[Dict[str, str]]:
    """Create synthetic edges for testing."""
    edges = []
    for i in range(n):
        edges.append({
            "src": f"account_{i % num_accounts}",
            "dst": f"account_{(i + 1) % num_accounts}",
            "amount": float(i * 10),
        })
    return edges


class TestBatchEdges:
    """Test batch_edges function."""
    
    def test_batch_edges_with_list(self):
        """Test batching a list of edges."""
        edges = _make_edges(25)
        batches = list(batch_edges(edges, batch_size=10))
        
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
    
    def test_batch_edges_with_iterator(self):
        """Test batching an iterator of edges."""
        edges = ({"src": f"a{i}", "dst": f"b{i}"} for i in range(25))
        batches = list(batch_edges(edges, batch_size=10))
        
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 25
    
    def test_batch_edges_empty(self):
        """Test batching empty edge list."""
        batches = list(batch_edges([], batch_size=10))
        assert len(batches) == 0
    
    def test_batch_edges_single_batch(self):
        """Test when all edges fit in single batch."""
        edges = _make_edges(5)
        batches = list(batch_edges(edges, batch_size=10))
        
        assert len(batches) == 1
        assert len(batches[0]) == 5


class TestBuildAdjacencyMatrixSparse:
    """Test sparse adjacency matrix construction."""
    
    def test_build_adjacency_directed(self):
        """Test building directed adjacency matrix."""
        edges = [
            {"src": "A", "dst": "B", "amount": 1.0},
            {"src": "B", "dst": "C", "amount": 2.0},
            {"src": "A", "dst": "C", "amount": 3.0},
        ]
        
        adj_matrix, node_index = build_adjacency_matrix_sparse(edges, directed=True)
        
        assert adj_matrix.shape == (3, 3)
        assert len(node_index) == 3
        assert adj_matrix.nnz == 3  # 3 directed edges
    
    def test_build_adjacency_undirected(self):
        """Test building undirected adjacency matrix."""
        edges = [
            {"src": "A", "dst": "B", "amount": 1.0},
            {"src": "B", "dst": "C", "amount": 2.0},
        ]
        
        adj_matrix, node_index = build_adjacency_matrix_sparse(edges, directed=False)
        
        assert adj_matrix.shape == (3, 3)
        assert adj_matrix.nnz == 4  # 2 edges * 2 for undirected
    
    def test_build_adjacency_with_node_index(self):
        """Test building with pre-computed node index."""
        edges = [
            {"src": "A", "dst": "B", "amount": 1.0},
        ]
        node_index = {"A": 0, "B": 1, "C": 2}
        
        adj_matrix, returned_index = build_adjacency_matrix_sparse(
            edges, node_index=node_index
        )
        
        assert returned_index == node_index
        assert adj_matrix.shape == (3, 3)
    
    def test_build_adjacency_empty(self):
        """Test building with empty edges."""
        adj_matrix, node_index = build_adjacency_matrix_sparse([])
        
        assert adj_matrix.shape == (0, 0)
        assert len(node_index) == 0


class TestGetRecommendedBatchSize:
    """Test batch size recommendations."""
    
    def test_small_graph(self):
        """Test recommendation for small graph."""
        assert get_recommended_batch_size(5_000) == BATCH_SIZE_RECOMMENDATIONS["small"]
    
    def test_medium_graph(self):
        """Test recommendation for medium graph."""
        assert get_recommended_batch_size(50_000) == BATCH_SIZE_RECOMMENDATIONS["medium"]
    
    def test_large_graph(self):
        """Test recommendation for large graph."""
        assert get_recommended_batch_size(500_000) == BATCH_SIZE_RECOMMENDATIONS["large"]
    
    def test_xlarge_graph(self):
        """Test recommendation for very large graph."""
        assert get_recommended_batch_size(2_000_000) == BATCH_SIZE_RECOMMENDATIONS["xlarge"]


class TestBatchGraphProcessor:
    """Test BatchGraphProcessor class."""
    
    def test_process_edges_without_progress(self):
        """Test processing edges without progress bar."""
        edges = _make_edges(100)
        processor = BatchGraphProcessor(batch_size=25, show_progress=False)
        
        results = []
        def process_fn(batch):
            results.append(len(batch))
            return len(batch)
        
        processor.process_edges(edges, process_fn)
        
        assert results == [25, 25, 25, 25]
    
    def test_process_edges_with_custom_batch_size(self):
        """Test processing with custom batch size."""
        edges = _make_edges(100)
        processor = BatchGraphProcessor(batch_size=30, show_progress=False)
        
        results = []
        def process_fn(batch):
            results.append(len(batch))
            return len(batch)
        
        processor.process_edges(edges, process_fn)
        
        assert results == [30, 30, 30, 10]


class TestComputeDegreeCentralityBatched:
    """Test batched degree centrality computation."""
    
    def test_compute_degree_centrality_batched_unweighted(self):
        """Test unweighted degree centrality with batching."""
        edges = [
            {"src": "A", "dst": "B"},
            {"src": "A", "dst": "C"},
            {"src": "B", "dst": "C"},
            {"src": "C", "dst": "A"},
        ]
        
        result = compute_degree_centrality_batched(
            edges, batch_size=2, show_progress=False, weighted=False
        )
        
        assert "A" in result.index
        assert "B" in result.index
        assert "C" in result.index
        # A has 3 connections (B, C, C->A)
        assert result["A"] == 3 / 2  # normalized by n-1
    
    def test_compute_degree_centrality_batched_weighted(self):
        """Test weighted degree centrality with batching."""
        edges = [
            {"src": "A", "dst": "B", "amount": 10.0},
            {"src": "A", "dst": "C", "amount": 20.0},
            {"src": "B", "dst": "C", "amount": 5.0},
        ]
        
        result = compute_degree_centrality_batched(
            edges, batch_size=2, show_progress=False, weighted=True
        )
        
        assert "A" in result.index
        # A sent 30, received 0 from these edges
        assert result["A"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
