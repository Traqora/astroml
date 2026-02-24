"""Graph integrity validation utilities.

This module provides validation and sanity checks for graph structures,
ensuring data quality before training ML models. It checks for isolated nodes,
edge consistency, and provides summary statistics.
"""
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import pandas as pd


class GraphValidationError(Exception):
    """Raised when graph validation fails critically."""
    pass


class GraphValidationWarning(UserWarning):
    """Warning for non-critical graph validation issues."""
    pass


def check_isolated_nodes(
    edges: pd.DataFrame,
    all_nodes: Optional[Set[str]] = None,
    source_col: str = "source",
    target_col: str = "target",
    allow_isolated: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """Check for isolated nodes in the graph.

    Args:
        edges: DataFrame containing edge list with source and target columns.
        all_nodes: Optional set of all nodes that should exist in the graph.
            If None, only nodes appearing in edges are considered.
        source_col: Name of the source node column.
        target_col: Name of the target node column.
        allow_isolated: If False, raises an error when isolated nodes are found.

    Returns:
        Tuple of (connected_nodes, isolated_nodes).

    Raises:
        GraphValidationError: If isolated nodes exist and allow_isolated=False.
        KeyError: If required columns are missing.

    Examples:
        >>> edges = pd.DataFrame({
        ...     "source": ["A", "B"],
        ...     "target": ["B", "C"]
        ... })
        >>> connected, isolated = check_isolated_nodes(edges, all_nodes={"A", "B", "C", "D"})
        >>> isolated
        {'D'}
    """
    if source_col not in edges.columns or target_col not in edges.columns:
        raise KeyError(f"DataFrame must contain '{source_col}' and '{target_col}' columns")

    # Get all nodes that appear in edges
    connected_nodes = set(edges[source_col].unique()) | set(edges[target_col].unique())

    # Determine isolated nodes
    if all_nodes is not None:
        isolated_nodes = all_nodes - connected_nodes
    else:
        isolated_nodes = set()

    if isolated_nodes and not allow_isolated:
        raise GraphValidationError(
            f"Found {len(isolated_nodes)} isolated nodes: {isolated_nodes}"
        )
    elif isolated_nodes:
        warnings.warn(
            f"Found {len(isolated_nodes)} isolated nodes",
            GraphValidationWarning
        )

    return connected_nodes, isolated_nodes


def check_edge_consistency(
    edges: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    allow_self_loops: bool = True,
    allow_duplicates: bool = False,
) -> Dict[str, Union[int, List[Tuple]]]:
    """Validate edge consistency in the graph.

    Args:
        edges: DataFrame containing edge list.
        source_col: Name of the source node column.
        target_col: Name of the target node column.
        weight_col: Optional name of edge weight column to check for validity.
        allow_self_loops: If False, raises error when self-loops are found.
        allow_duplicates: If False, raises error when duplicate edges exist.

    Returns:
        Dictionary with validation results:
            - 'self_loops': count of self-loop edges
            - 'duplicate_edges': count of duplicate edges
            - 'negative_weights': count of negative weights (if weight_col provided)
            - 'null_values': count of null values in edge columns

    Raises:
        GraphValidationError: If validation fails based on parameters.
        KeyError: If required columns are missing.

    Examples:
        >>> edges = pd.DataFrame({
        ...     "source": ["A", "B", "A"],
        ...     "target": ["A", "C", "A"]
        ... })
        >>> result = check_edge_consistency(edges, allow_self_loops=True)
        >>> result['self_loops']
        2
    """
    if source_col not in edges.columns or target_col not in edges.columns:
        raise KeyError(f"DataFrame must contain '{source_col}' and '{target_col}' columns")

    results = {}

    # Check for null values
    null_sources = edges[source_col].isnull().sum()
    null_targets = edges[target_col].isnull().sum()
    results['null_values'] = int(null_sources + null_targets)

    if results['null_values'] > 0:
        raise GraphValidationError(
            f"Found {results['null_values']} null values in edge columns"
        )

    # Check for self-loops
    self_loop_mask = edges[source_col] == edges[target_col]
    results['self_loops'] = int(self_loop_mask.sum())

    if results['self_loops'] > 0 and not allow_self_loops:
        raise GraphValidationError(
            f"Found {results['self_loops']} self-loop edges (not allowed)"
        )

    # Check for duplicate edges
    edge_pairs = edges[[source_col, target_col]]
    duplicates = edge_pairs.duplicated()
    results['duplicate_edges'] = int(duplicates.sum())

    if results['duplicate_edges'] > 0 and not allow_duplicates:
        raise GraphValidationError(
            f"Found {results['duplicate_edges']} duplicate edges (not allowed)"
        )

    # Check weights if provided
    if weight_col is not None:
        if weight_col not in edges.columns:
            raise KeyError(f"Weight column '{weight_col}' not found in DataFrame")

        negative_weights = (edges[weight_col] < 0).sum()
        results['negative_weights'] = int(negative_weights)

        if results['negative_weights'] > 0:
            warnings.warn(
                f"Found {results['negative_weights']} edges with negative weights",
                GraphValidationWarning
            )

    return results


def graph_summary_statistics(
    edges: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
) -> Dict[str, Union[int, float, Dict]]:
    """Generate summary statistics for the graph.

    Args:
        edges: DataFrame containing edge list.
        source_col: Name of the source node column.
        target_col: Name of the target node column.
        weight_col: Optional name of edge weight column for weight statistics.

    Returns:
        Dictionary containing:
            - 'num_edges': Total number of edges
            - 'num_nodes': Total number of unique nodes
            - 'num_source_nodes': Number of unique source nodes
            - 'num_target_nodes': Number of unique target nodes
            - 'density': Graph density (edges / possible_edges)
            - 'avg_degree': Average node degree
            - 'degree_stats': Dict with min, max, median, std of degrees
            - 'weight_stats': Dict with weight statistics (if weight_col provided)

    Examples:
        >>> edges = pd.DataFrame({
        ...     "source": ["A", "B", "C"],
        ...     "target": ["B", "C", "A"]
        ... })
        >>> stats = graph_summary_statistics(edges)
        >>> stats['num_nodes']
        3
    """
    if source_col not in edges.columns or target_col not in edges.columns:
        raise KeyError(f"DataFrame must contain '{source_col}' and '{target_col}' columns")

    stats = {}

    # Basic counts
    stats['num_edges'] = len(edges)
    source_nodes = set(edges[source_col].unique())
    target_nodes = set(edges[target_col].unique())
    all_nodes = source_nodes | target_nodes

    stats['num_nodes'] = len(all_nodes)
    stats['num_source_nodes'] = len(source_nodes)
    stats['num_target_nodes'] = len(target_nodes)

    # Graph density
    num_nodes = stats['num_nodes']
    max_edges = num_nodes * (num_nodes - 1)  # directed graph
    stats['density'] = stats['num_edges'] / max_edges if max_edges > 0 else 0.0

    # Degree statistics
    out_degrees = edges[source_col].value_counts()
    in_degrees = edges[target_col].value_counts()

    # Total degree for each node
    all_degrees = pd.Series(0, index=list(all_nodes))
    all_degrees = all_degrees.add(out_degrees, fill_value=0).add(in_degrees, fill_value=0)

    stats['avg_degree'] = float(all_degrees.mean())
    stats['degree_stats'] = {
        'min': int(all_degrees.min()),
        'max': int(all_degrees.max()),
        'median': float(all_degrees.median()),
        'std': float(all_degrees.std()),
    }

    # Weight statistics if provided
    if weight_col is not None:
        if weight_col not in edges.columns:
            raise KeyError(f"Weight column '{weight_col}' not found in DataFrame")

        weights = edges[weight_col]
        stats['weight_stats'] = {
            'min': float(weights.min()),
            'max': float(weights.max()),
            'mean': float(weights.mean()),
            'median': float(weights.median()),
            'std': float(weights.std()),
            'sum': float(weights.sum()),
        }

    return stats


def validate_graph(
    edges: pd.DataFrame,
    all_nodes: Optional[Set[str]] = None,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    allow_isolated: bool = False,
    allow_self_loops: bool = True,
    allow_duplicates: bool = False,
    verbose: bool = True,
) -> Dict[str, Union[Dict, Set]]:
    """Comprehensive graph validation with all checks.

    This is a convenience function that runs all validation checks and
    returns a comprehensive report.

    Args:
        edges: DataFrame containing edge list.
        all_nodes: Optional set of all nodes that should exist.
        source_col: Name of the source node column.
        target_col: Name of the target node column.
        weight_col: Optional name of edge weight column.
        allow_isolated: If False, raises error for isolated nodes.
        allow_self_loops: If False, raises error for self-loops.
        allow_duplicates: If False, raises error for duplicate edges.
        verbose: If True, prints validation summary.

    Returns:
        Dictionary containing:
            - 'summary': Summary statistics from graph_summary_statistics
            - 'isolated_nodes': Set of isolated nodes
            - 'edge_checks': Results from check_edge_consistency
            - 'validation_passed': Boolean indicating if all checks passed

    Raises:
        GraphValidationError: If any validation check fails.

    Examples:
        >>> edges = pd.DataFrame({
        ...     "source": ["A", "B"],
        ...     "target": ["B", "C"]
        ... })
        >>> report = validate_graph(edges, verbose=False)
        >>> report['validation_passed']
        True
    """
    report = {}

    # Run all checks
    connected_nodes, isolated_nodes = check_isolated_nodes(
        edges, all_nodes, source_col, target_col, allow_isolated
    )
    report['isolated_nodes'] = isolated_nodes

    edge_checks = check_edge_consistency(
        edges, source_col, target_col, weight_col,
        allow_self_loops, allow_duplicates
    )
    report['edge_checks'] = edge_checks

    summary = graph_summary_statistics(edges, source_col, target_col, weight_col)
    report['summary'] = summary

    # Determine if validation passed
    validation_passed = (
        (len(isolated_nodes) == 0 or allow_isolated) and
        (edge_checks['self_loops'] == 0 or allow_self_loops) and
        (edge_checks['duplicate_edges'] == 0 or allow_duplicates) and
        edge_checks['null_values'] == 0
    )
    report['validation_passed'] = validation_passed

    if verbose:
        print("=" * 60)
        print("GRAPH VALIDATION REPORT")
        print("=" * 60)
        print(f"\nNodes: {summary['num_nodes']}")
        print(f"Edges: {summary['num_edges']}")
        print(f"Density: {summary['density']:.6f}")
        print(f"Average Degree: {summary['avg_degree']:.2f}")
        print(f"\nDegree Statistics:")
        print(f"  Min: {summary['degree_stats']['min']}")
        print(f"  Max: {summary['degree_stats']['max']}")
        print(f"  Median: {summary['degree_stats']['median']:.2f}")
        print(f"  Std: {summary['degree_stats']['std']:.2f}")

        if weight_col and 'weight_stats' in summary:
            print(f"\nWeight Statistics:")
            print(f"  Min: {summary['weight_stats']['min']:.2f}")
            print(f"  Max: {summary['weight_stats']['max']:.2f}")
            print(f"  Mean: {summary['weight_stats']['mean']:.2f}")
            print(f"  Sum: {summary['weight_stats']['sum']:.2f}")

        print(f"\nEdge Checks:")
        print(f"  Self-loops: {edge_checks['self_loops']}")
        print(f"  Duplicate edges: {edge_checks['duplicate_edges']}")
        print(f"  Null values: {edge_checks['null_values']}")

        if isolated_nodes:
            print(f"\nIsolated Nodes: {len(isolated_nodes)}")

        print(f"\nValidation Status: {'PASSED' if validation_passed else 'FAILED'}")
        print("=" * 60)

    return report
