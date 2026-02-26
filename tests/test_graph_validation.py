import numpy as np
import pandas as pd
import pytest

from astroml.features import graph_validation


def test_check_isolated_nodes_no_isolated():
    """Test when all nodes are connected."""
    edges = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "A"]
    })
    connected, isolated = graph_validation.check_isolated_nodes(edges)
    assert len(isolated) == 0
    assert connected == {"A", "B", "C"}


def test_check_isolated_nodes_with_isolated():
    """Test detection of isolated nodes."""
    edges = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "C"]
    })
    all_nodes = {"A", "B", "C", "D", "E"}
    connected, isolated = graph_validation.check_isolated_nodes(
        edges, all_nodes=all_nodes, allow_isolated=True
    )
    assert isolated == {"D", "E"}
    assert connected == {"A", "B", "C"}


def test_check_isolated_nodes_raises_error():
    """Test that isolated nodes raise error when not allowed."""
    edges = pd.DataFrame({
        "source": ["A"],
        "target": ["B"]
    })
    all_nodes = {"A", "B", "C"}

    with pytest.raises(graph_validation.GraphValidationError):
        graph_validation.check_isolated_nodes(
            edges, all_nodes=all_nodes, allow_isolated=False
        )


def test_check_edge_consistency_self_loops():
    """Test detection of self-loops."""
    edges = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["A", "C", "D"]
    })
    result = graph_validation.check_edge_consistency(edges, allow_self_loops=True)
    assert result['self_loops'] == 1


def test_check_edge_consistency_self_loops_not_allowed():
    """Test that self-loops raise error when not allowed."""
    edges = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["A", "C"]
    })

    with pytest.raises(graph_validation.GraphValidationError):
        graph_validation.check_edge_consistency(edges, allow_self_loops=False)


def test_check_edge_consistency_duplicates():
    """Test detection of duplicate edges."""
    edges = pd.DataFrame({
        "source": ["A", "B", "A"],
        "target": ["B", "C", "B"]
    })
    result = graph_validation.check_edge_consistency(edges, allow_duplicates=True)
    assert result['duplicate_edges'] == 1


def test_check_edge_consistency_duplicates_not_allowed():
    """Test that duplicates raise error when not allowed."""
    edges = pd.DataFrame({
        "source": ["A", "A"],
        "target": ["B", "B"]
    })

    with pytest.raises(graph_validation.GraphValidationError):
        graph_validation.check_edge_consistency(edges, allow_duplicates=False)


def test_check_edge_consistency_null_values():
    """Test that null values raise error."""
    edges = pd.DataFrame({
        "source": ["A", None, "C"],
        "target": ["B", "C", "D"]
    })

    with pytest.raises(graph_validation.GraphValidationError):
        graph_validation.check_edge_consistency(edges)


def test_check_edge_consistency_negative_weights():
    """Test detection of negative weights."""
    edges = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "D"],
        "weight": [1.0, -2.0, 3.0]
    })

    with pytest.warns(graph_validation.GraphValidationWarning):
        result = graph_validation.check_edge_consistency(edges, weight_col="weight")
        assert result['negative_weights'] == 1


def test_graph_summary_statistics_basic():
    """Test basic summary statistics."""
    edges = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "A"]
    })
    stats = graph_validation.graph_summary_statistics(edges)

    assert stats['num_edges'] == 3
    assert stats['num_nodes'] == 3
    assert stats['num_source_nodes'] == 3
    assert stats['num_target_nodes'] == 3
    assert stats['avg_degree'] == 2.0


def test_graph_summary_statistics_with_weights():
    """Test summary statistics with edge weights."""
    edges = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "C"],
        "weight": [10.0, 20.0]
    })
    stats = graph_validation.graph_summary_statistics(edges, weight_col="weight")

    assert 'weight_stats' in stats
    assert stats['weight_stats']['min'] == 10.0
    assert stats['weight_stats']['max'] == 20.0
    assert stats['weight_stats']['mean'] == 15.0
    assert stats['weight_stats']['sum'] == 30.0


def test_graph_summary_statistics_degree_stats():
    """Test degree statistics calculation."""
    edges = pd.DataFrame({
        "source": ["A", "A", "B"],
        "target": ["B", "C", "C"]
    })
    stats = graph_validation.graph_summary_statistics(edges)

    # Node A: out-degree 2, in-degree 0 = total 2
    # Node B: out-degree 1, in-degree 1 = total 2
    # Node C: out-degree 0, in-degree 2 = total 2
    assert stats['degree_stats']['min'] == 2
    assert stats['degree_stats']['max'] == 2
    assert stats['degree_stats']['median'] == 2.0


def test_graph_summary_statistics_density():
    """Test graph density calculation."""
    edges = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "A"]
    })
    stats = graph_validation.graph_summary_statistics(edges)

    # 2 nodes, max edges = 2 * (2-1) = 2, actual edges = 2
    assert stats['density'] == 1.0


def test_validate_graph_comprehensive():
    """Test comprehensive validation with all checks."""
    edges = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "D"],
        "weight": [1.0, 2.0, 3.0]
    })

    report = graph_validation.validate_graph(
        edges,
        weight_col="weight",
        verbose=False
    )

    assert report['validation_passed'] is True
    assert len(report['isolated_nodes']) == 0
    assert 'summary' in report
    assert 'edge_checks' in report


def test_validate_graph_with_issues():
    """Test validation with multiple issues."""
    edges = pd.DataFrame({
        "source": ["A", "A", "B"],
        "target": ["A", "A", "C"]
    })

    report = graph_validation.validate_graph(
        edges,
        allow_self_loops=True,
        allow_duplicates=True,
        verbose=False
    )

    assert report['edge_checks']['self_loops'] == 2
    assert report['edge_checks']['duplicate_edges'] == 1


def test_custom_column_names():
    """Test with custom column names."""
    edges = pd.DataFrame({
        "from": ["A", "B"],
        "to": ["B", "C"],
        "amount": [100, 200]
    })

    stats = graph_validation.graph_summary_statistics(
        edges,
        source_col="from",
        target_col="to",
        weight_col="amount"
    )

    assert stats['num_edges'] == 2
    assert stats['weight_stats']['sum'] == 300.0


def test_missing_columns_raises_error():
    """Test that missing columns raise appropriate errors."""
    edges = pd.DataFrame({
        "source": ["A", "B"],
        "wrong_col": ["B", "C"]
    })

    with pytest.raises(KeyError):
        graph_validation.check_isolated_nodes(edges)

    with pytest.raises(KeyError):
        graph_validation.check_edge_consistency(edges)

    with pytest.raises(KeyError):
        graph_validation.graph_summary_statistics(edges)


def test_empty_graph():
    """Test handling of empty graph."""
    edges = pd.DataFrame({
        "source": [],
        "target": []
    })

    stats = graph_validation.graph_summary_statistics(edges)
    assert stats['num_edges'] == 0
    assert stats['num_nodes'] == 0
    assert stats['density'] == 0.0
