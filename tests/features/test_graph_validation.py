"""Comprehensive tests for graph validation logic.

Tests graph validation functions in astroml/features/graph_validation.py:
- Valid graph passes all checks
- Graph with self-loops detected
- Graph with isolated nodes detected
- Graph with disconnected components detected
- Graph with negative edge weights rejected
- Multi-edge handling (deduplication)
- NetworkX test graphs (KarateClub, DavisSouthernWomen)
- Large graph validation (10k nodes, 100k edges)
"""

from __future__ import annotations

import warnings

import networkx as nx
import pandas as pd
import pytest

from astroml.features.graph_validation import (
    GraphValidationError,
    GraphValidationWarning,
    check_edge_consistency,
    check_isolated_nodes,
    graph_summary_statistics,
    validate_graph,
)


class TestGraphValidation:
    """Test suite for graph validation functions."""

    @pytest.fixture
    def valid_graph_edges(self):
        """Create a valid simple graph for testing."""
        return pd.DataFrame(
            {
                "source": ["A", "B", "C", "D"],
                "target": ["B", "C", "D", "A"],
                "weight": [1.0, 2.0, 3.0, 4.0],
            }
        )

    @pytest.fixture
    def graph_with_self_loops(self):
        """Create a graph with self-loops."""
        return pd.DataFrame(
            {
                "source": ["A", "B", "A", "C"],
                "target": ["B", "C", "A", "D"],
            }
        )

    @pytest.fixture
    def graph_with_duplicates(self):
        """Create a graph with duplicate edges."""
        return pd.DataFrame(
            {
                "source": ["A", "B", "A", "B"],
                "target": ["B", "C", "B", "C"],
            }
        )

    @pytest.fixture
    def graph_with_negative_weights(self):
        """Create a graph with negative edge weights."""
        return pd.DataFrame(
            {
                "source": ["A", "B", "C"],
                "target": ["B", "C", "A"],
                "weight": [1.0, -2.0, 3.0],
            }
        )

    def test_valid_graph_passes_all_checks(self, valid_graph_edges):
        """Test that a valid graph passes all validation checks."""
        # Check isolated nodes
        connected, isolated = check_isolated_nodes(
            valid_graph_edges, all_nodes={"A", "B", "C", "D"}, allow_isolated=False
        )
        assert len(isolated) == 0
        assert len(connected) == 4

        # Check edge consistency
        edge_checks = check_edge_consistency(
            valid_graph_edges, weight_col="weight", allow_self_loops=True, allow_duplicates=False
        )
        assert edge_checks["self_loops"] == 0
        assert edge_checks["duplicate_edges"] == 0
        assert edge_checks["null_values"] == 0
        assert edge_checks["negative_weights"] == 0

        # Validate graph
        report = validate_graph(
            valid_graph_edges, all_nodes={"A", "B", "C", "D"}, weight_col="weight", verbose=False
        )
        assert report["validation_passed"] is True

    def test_graph_with_self_loops_detected(self, graph_with_self_loops):
        """Test that self-loops are detected correctly."""
        # With self-loops allowed
        edge_checks = check_edge_consistency(graph_with_self_loops, allow_self_loops=True)
        assert edge_checks["self_loops"] == 1

        # With self-loops not allowed
        with pytest.raises(GraphValidationError, match="self-loop"):
            check_edge_consistency(graph_with_self_loops, allow_self_loops=False)

    def test_graph_with_isolated_nodes_detected(self, valid_graph_edges):
        """Test that isolated nodes are detected correctly."""
        # Create graph with isolated nodes
        all_nodes = {"A", "B", "C", "D", "E", "F"}

        connected, isolated = check_isolated_nodes(
            valid_graph_edges, all_nodes=all_nodes, allow_isolated=True
        )

        assert len(isolated) == 2  # E and F are isolated
        assert "E" in isolated
        assert "F" in isolated
        assert len(connected) == 4

    def test_isolated_nodes_raise_error_when_not_allowed(self, valid_graph_edges):
        """Test that isolated nodes raise error when not allowed."""
        all_nodes = {"A", "B", "C", "D", "E"}

        with pytest.raises(GraphValidationError, match="isolated nodes"):
            check_isolated_nodes(valid_graph_edges, all_nodes=all_nodes, allow_isolated=False)

    def test_disconnected_components_detected(self):
        """Test detection of disconnected graph components."""
        # Create graph with two disconnected components
        edges = pd.DataFrame(
            {
                "source": ["A", "B", "X", "Y"],
                "target": ["B", "A", "Y", "X"],
            }
        )

        connected, isolated = check_isolated_nodes(
            edges, all_nodes={"A", "B", "X", "Y"}, allow_isolated=True
        )

        # All nodes are connected within their components
        assert len(isolated) == 0
        assert len(connected) == 4

        # But the graph has 2 components (A-B and X-Y)
        # This can be verified by checking connectivity
        G = nx.from_pandas_edgelist(edges, source="source", target="target")
        assert nx.number_connected_components(G) == 2

    def test_negative_edge_weights_rejected(self, graph_with_negative_weights):
        """Test that negative edge weights are detected and warned."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            edge_checks = check_edge_consistency(
                graph_with_negative_weights, weight_col="weight", allow_self_loops=True
            )

            # Should have warning about negative weights
            assert len(w) == 1
            assert issubclass(w[0].category, GraphValidationWarning)
            assert "negative weights" in str(w[0].message)

        assert edge_checks["negative_weights"] == 1

    def test_multi_edge_deduplication(self, graph_with_duplicates):
        """Test that duplicate edges are detected and handled."""
        # With duplicates allowed
        edge_checks = check_edge_consistency(graph_with_duplicates, allow_duplicates=True)
        assert edge_checks["duplicate_edges"] == 2  # A->B and B->C are duplicated

        # With duplicates not allowed
        with pytest.raises(GraphValidationError, match="duplicate edges"):
            check_edge_consistency(graph_with_duplicates, allow_duplicates=False)

    def test_networkx_karate_club_graph(self):
        """Test validation with NetworkX Karate Club graph."""
        G = nx.karate_club_graph()

        # Convert to edge list
        edges_list = [(u, v) for u, v in G.edges()]
        edges_df = pd.DataFrame(edges_list, columns=["source", "target"])

        # Validate graph
        report = validate_graph(edges_df, verbose=False)

        assert report["validation_passed"] is True
        assert report["summary"]["num_nodes"] == G.number_of_nodes()
        assert report["summary"]["num_edges"] == G.number_of_edges()
        assert report["edge_checks"]["self_loops"] == 0
        assert report["edge_checks"]["duplicate_edges"] == 0

    def test_networkx_davis_southern_women_graph(self):
        """Test validation with NetworkX Davis Southern Women graph."""
        G = nx.davis_southern_women_graph()

        # Convert to edge list (bipartite graph)
        edges_list = [(u, v) for u, v in G.edges()]
        edges_df = pd.DataFrame(edges_list, columns=["source", "target"])

        # Validate graph
        report = validate_graph(edges_df, verbose=False)

        assert report["validation_passed"] is True
        assert report["summary"]["num_nodes"] == G.number_of_nodes()
        assert report["summary"]["num_edges"] == G.number_of_edges()

    def test_large_graph_validation(self):
        """Test validation on large graph (10k nodes, 100k edges)."""
        # Create large random graph
        num_nodes = 10000
        num_edges = 100000

        # Generate random edges
        import numpy as np

        sources = np.random.randint(0, num_nodes, num_edges)
        targets = np.random.randint(0, num_nodes, num_edges)

        edges_df = pd.DataFrame(
            {
                "source": [f"node_{s}" for s in sources],
                "target": [f"node_{t}" for t in targets],
            }
        )

        # Validate graph
        report = validate_graph(edges_df, verbose=False)

        assert report["validation_passed"] is True
        assert report["summary"]["num_edges"] == num_edges
        assert report["edge_checks"]["null_values"] == 0

    def test_null_values_in_edges(self):
        """Test that null values in edge columns are detected."""
        edges = pd.DataFrame(
            {
                "source": ["A", "B", None, "D"],
                "target": ["B", "C", "D", "A"],
            }
        )

        with pytest.raises(GraphValidationError, match="null values"):
            check_edge_consistency(edges)

    def test_missing_columns_raise_error(self):
        """Test that missing required columns raise KeyError."""
        edges = pd.DataFrame(
            {
                "source": ["A", "B"],
                "wrong_column": ["B", "C"],
            }
        )

        with pytest.raises(KeyError, match="target"):
            check_isolated_nodes(edges)

        with pytest.raises(KeyError, match="target"):
            check_edge_consistency(edges)

    def test_empty_graph(self):
        """Test validation of empty graph."""
        edges = pd.DataFrame(
            {
                "source": [],
                "target": [],
            }
        )

        connected, isolated = check_isolated_nodes(edges)
        assert len(connected) == 0
        assert len(isolated) == 0

        edge_checks = check_edge_consistency(edges)
        assert edge_checks["self_loops"] == 0
        assert edge_checks["duplicate_edges"] == 0

    def test_graph_summary_statistics(self, valid_graph_edges):
        """Test graph summary statistics calculation."""
        stats = graph_summary_statistics(valid_graph_edges, weight_col="weight")

        assert stats["num_edges"] == 4
        assert stats["num_nodes"] == 4
        assert stats["num_source_nodes"] == 4
        assert stats["num_target_nodes"] == 4
        assert stats["density"] > 0
        assert stats["avg_degree"] > 0

        # Check degree stats
        assert "min" in stats["degree_stats"]
        assert "max" in stats["degree_stats"]
        assert "median" in stats["degree_stats"]
        assert "std" in stats["degree_stats"]

        # Check weight stats
        assert "min" in stats["weight_stats"]
        assert "max" in stats["weight_stats"]
        assert "mean" in stats["weight_stats"]
        assert stats["weight_stats"]["sum"] == 10.0

    def test_validate_graph_comprehensive(self, valid_graph_edges):
        """Test comprehensive validation function."""
        report = validate_graph(
            valid_graph_edges, all_nodes={"A", "B", "C", "D"}, weight_col="weight", verbose=False
        )

        assert "summary" in report
        assert "isolated_nodes" in report
        assert "edge_checks" in report
        assert "validation_passed" in report

        assert report["validation_passed"] is True
        assert len(report["isolated_nodes"]) == 0

    def test_validate_graph_with_warnings(self, graph_with_negative_weights):
        """Test validate_graph with warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            report = validate_graph(graph_with_negative_weights, weight_col="weight", verbose=False)

            # Should still pass but with warning
            assert report["validation_passed"] is True
            assert len(w) > 0

    def test_validate_graph_failure(self, graph_with_self_loops):
        """Test validate_graph failure when checks fail."""
        with pytest.raises(GraphValidationError):
            validate_graph(graph_with_self_loops, allow_self_loops=False, verbose=False)
