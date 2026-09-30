"""Tests for structural importance metrics module."""

import pandas as pd
import pytest

from astroml.features.structural_importance import (
    compute_betweenness_centrality,
    compute_closeness_centrality,
    compute_clustering_coefficient,
    compute_degree_centrality,
    compute_eigenvector_centrality,
    compute_pagerank,
    compute_strength_centrality,
    compute_structural_importance_metrics,
)


class TestStructuralImportance:
    """Test cases for structural importance metrics."""

    def setup_method(self):
        """Set up test data."""
        # Simple test graph
        self.edges = [
            {"src": "A", "dst": "B", "amount": 10.0, "timestamp": 1000},
            {"src": "A", "dst": "C", "amount": 5.0, "timestamp": 1001},
            {"src": "B", "dst": "C", "amount": 7.0, "timestamp": 1002},
            {"src": "C", "dst": "A", "amount": 3.0, "timestamp": 1003},
            {"src": "D", "dst": "A", "amount": 8.0, "timestamp": 1004},
        ]

        # Star graph for testing
        self.star_edges = [
            {"src": "center", "dst": "leaf1", "amount": 1.0, "timestamp": 1000},
            {"src": "center", "dst": "leaf2", "amount": 1.0, "timestamp": 1001},
            {"src": "center", "dst": "leaf3", "amount": 1.0, "timestamp": 1002},
        ]

        # Complete graph for testing
        self.complete_edges = [
            {"src": "1", "dst": "2", "amount": 1.0, "timestamp": 1000},
            {"src": "1", "dst": "3", "amount": 1.0, "timestamp": 1001},
            {"src": "2", "dst": "1", "amount": 1.0, "timestamp": 1002},
            {"src": "2", "dst": "3", "amount": 1.0, "timestamp": 1003},
            {"src": "3", "dst": "1", "amount": 1.0, "timestamp": 1004},
            {"src": "3", "dst": "2", "amount": 1.0, "timestamp": 1005},
        ]

    def test_degree_centrality(self):
        """Test degree centrality computation."""
        # Unweighted degree
        result = compute_degree_centrality(self.edges, weighted=False)

        expected_nodes = {"A", "B", "C", "D"}
        assert set(result.index) == expected_nodes

        # A has degree 3 (edges to B, C, and from C, D)
        # B has degree 2 (edge from A, to C)
        # C has degree 3 (edges from A, B, to A)
        # D has degree 1 (edge to A)
        assert result["A"] == pytest.approx(1.0)  # 3/(4-1) = 1.0
        assert result["B"] == pytest.approx(2 / 3)  # 2/(4-1) = 2/3
        assert result["C"] == pytest.approx(1.0)  # 3/(4-1) = 1.0
        assert result["D"] == pytest.approx(1 / 3)  # 1/(4-1) = 1/3

        # Weighted degree
        result_weighted = compute_degree_centrality(self.edges, weighted=True)
        assert result_weighted["A"] > result_weighted["D"]  # A has more volume

    def test_strength_centrality(self):
        """Test strength centrality computation."""
        result = compute_strength_centrality(self.edges, direction="both")

        expected_nodes = {"A", "B", "C", "D"}
        assert set(result.index) == expected_nodes

        # A: receives 8.0 from D, sends 15.0 (10+5), receives 3.0 from C = 26.0
        # B: receives 10.0 from A, sends 7.0 to C = 17.0
        # C: receives 5.0 from A, 7.0 from B, sends 3.0 to A = 15.0
        # D: sends 8.0 to A = 8.0
        assert result["A"] == pytest.approx(26.0)
        assert result["B"] == pytest.approx(17.0)
        assert result["C"] == pytest.approx(15.0)
        assert result["D"] == pytest.approx(8.0)

    def test_pagerank(self):
        """Test PageRank computation."""
        result = compute_pagerank(self.edges)

        expected_nodes = {"A", "B", "C", "D"}
        assert set(result.index) == expected_nodes

        # All values should be positive and sum to 1
        assert all(result > 0)
        assert pytest.approx(result.sum(), 1.0)

        # A should have higher PageRank due to more connections
        assert result["A"] > result["D"]

    def test_clustering_coefficient(self):
        """Test clustering coefficient computation."""
        # Complete graph should have clustering coefficient of 1
        result_complete = compute_clustering_coefficient(self.complete_edges)
        for node in result_complete.index:
            assert result_complete[node] == pytest.approx(1.0)

        # Star graph should have clustering coefficient of 0
        result_star = compute_clustering_coefficient(self.star_edges)
        for node in result_star.index:
            assert result_star[node] == pytest.approx(0.0)

    def test_betweenness_centrality(self):
        """Test betweenness centrality computation."""
        # Star graph: center node should have highest betweenness
        result = compute_betweenness_centrality(self.star_edges, normalized=True)

        expected_nodes = {"center", "leaf1", "leaf2", "leaf3"}
        assert set(result.index) == expected_nodes

        # Center should have highest betweenness
        assert result["center"] > result["leaf1"]
        assert result["center"] > result["leaf2"]
        assert result["center"] > result["leaf3"]

        # Leaves should have betweenness of 0
        assert result["leaf1"] == pytest.approx(0.0)
        assert result["leaf2"] == pytest.approx(0.0)
        assert result["leaf3"] == pytest.approx(0.0)

    def test_closeness_centrality(self):
        """Test closeness centrality computation."""
        # Complete graph should have equal closeness for all nodes
        result_complete = compute_closeness_centrality(self.complete_edges)

        # All nodes should have the same closeness
        values = result_complete.tolist()
        assert all(abs(v - values[0]) < 1e-10 for v in values)

    def test_comprehensive_metrics(self):
        """Test the comprehensive metrics function."""
        result = compute_structural_importance_metrics(
            self.edges,
            include_betweenness=True,
            include_closeness=True,
            include_eigenvector=False,  # Skip scipy-dependent test
        )

        expected_nodes = {"A", "B", "C", "D"}
        assert set(result.index) == expected_nodes

        expected_columns = {
            "degree_centrality",
            "strength_centrality",
            "pagerank",
            "clustering_coefficient",
            "betweenness_centrality",
            "closeness_centrality",
        }
        assert set(result.columns) == expected_columns

        # All values should be non-negative
        assert (result >= 0).all().all()

    def test_empty_edges(self):
        """Test handling of empty edge list."""
        result = compute_structural_importance_metrics([])

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_single_node(self):
        """Test handling of single node."""
        single_edge = [{"src": "A", "dst": "A", "amount": 1.0, "timestamp": 1000}]

        result = compute_structural_importance_metrics(single_edge)

        assert len(result) == 1
        assert "A" in result.index

    def test_node_filter(self):
        """Test node filtering functionality."""
        nodes = ["A", "B"]

        result = compute_degree_centrality(self.edges, nodes=nodes)

        assert set(result.index) == {"A", "B"}

    def test_edge_direction_handling(self):
        """Test that edge directions are handled correctly."""
        # Test with self-loops
        self_loop_edges = [{"src": "A", "dst": "A", "amount": 1.0, "timestamp": 1000}]

        result = compute_degree_centrality(self_loop_edges)
        assert "A" in result.index
        assert result["A"] >= 0

    @pytest.mark.skipif(
        not pytest.importorskip("scipy", reason="SciPy not available"),
        reason="SciPy required for eigenvector centrality",
    )
    def test_eigenvector_centrality(self):
        """Test eigenvector centrality computation (requires SciPy)."""
        result = compute_eigenvector_centrality(self.complete_edges)

        expected_nodes = {"1", "2", "3"}
        assert set(result.index) == expected_nodes

        # All values should be positive
        assert all(result > 0)


class TestPipelineIntegration:
    """Test cases for pipeline integration."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from astroml.features.pipeline_structural_importance import StructuralImportancePipeline

        pipeline = StructuralImportancePipeline(
            include_betweenness=False, include_closeness=False, pagerank_sample_size=100
        )

        assert pipeline.include_betweenness is False
        assert pipeline.include_closeness is False
        assert pipeline.pagerank_sample_size == 100

    def test_edge_list_processing(self):
        """Test processing edge lists directly."""
        from astroml.features.pipeline_structural_importance import StructuralImportancePipeline

        edges = [
            {"src": "A", "dst": "B", "amount": 10.0, "timestamp": 1000},
            {"src": "B", "dst": "C", "amount": 5.0, "timestamp": 1001},
        ]

        pipeline = StructuralImportancePipeline(include_betweenness=False)
        result = pipeline.process_edge_list(edges)

        assert set(result.index) == {"A", "B", "C"}
        assert "degree_centrality" in result.columns
        assert "pagerank" in result.columns

    def test_summary_statistics(self):
        """Test summary statistics generation."""
        from astroml.features.pipeline_structural_importance import StructuralImportancePipeline

        # Create test metrics
        metrics = pd.DataFrame(
            {"degree_centrality": [0.5, 0.3, 0.2], "pagerank": [0.4, 0.3, 0.3]},
            index=["A", "B", "C"],
        )

        pipeline = StructuralImportancePipeline()
        summary = pipeline.get_summary_statistics(metrics)

        assert summary["total_accounts"] == 3
        assert "degree_centrality" in summary["metrics"]
        assert "pagerank" in summary["metrics"]

        # Check metric statistics
        dc_stats = summary["metrics"]["degree_centrality"]
        assert dc_stats["mean"] == pytest.approx(0.3333333333333333)
        assert dc_stats["min"] == 0.2
        assert dc_stats["max"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
