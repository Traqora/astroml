"""Demo script for graph validation functionality.

This script demonstrates how to use the graph validation utilities
to check graph integrity before training ML models.
"""
import pandas as pd
from astroml.features import graph_validation


def demo_basic_validation():
    """Demonstrate basic graph validation."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Graph Validation")
    print("="*60)

    # Create a simple transaction graph
    edges = pd.DataFrame({
        "source": ["Alice", "Bob", "Charlie", "Alice"],
        "target": ["Bob", "Charlie", "Alice", "David"],
        "amount": [100.0, 50.0, 75.0, 200.0]
    })

    # Run comprehensive validation
    report = graph_validation.validate_graph(
        edges,
        source_col="source",
        target_col="target",
        weight_col="amount",
        verbose=True
    )

    print(f"\nValidation passed: {report['validation_passed']}")


def demo_isolated_nodes():
    """Demonstrate isolated node detection."""
    print("\n" + "="*60)
    print("DEMO 2: Isolated Node Detection")
    print("="*60)

    edges = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "C"]
    })

    # Define all nodes that should exist
    all_nodes = {"A", "B", "C", "D", "E"}

    print(f"\nExpected nodes: {all_nodes}")
    print(f"Edges: {len(edges)}")

    connected, isolated = graph_validation.check_isolated_nodes(
        edges,
        all_nodes=all_nodes,
        allow_isolated=True
    )

    print(f"\nConnected nodes: {connected}")
    print(f"Isolated nodes: {isolated}")


def demo_edge_consistency():
    """Demonstrate edge consistency checks."""
    print("\n" + "="*60)
    print("DEMO 3: Edge Consistency Checks")
    print("="*60)

    # Graph with various edge issues
    edges = pd.DataFrame({
        "source": ["A", "B", "C", "A", "D"],
        "target": ["A", "C", "D", "B", "D"],
        "weight": [10.0, 20.0, 30.0, 15.0, -5.0]
    })

    print("\nChecking edge consistency...")
    result = graph_validation.check_edge_consistency(
        edges,
        weight_col="weight",
        allow_self_loops=True,
        allow_duplicates=True
    )

    print(f"\nSelf-loops found: {result['self_loops']}")
    print(f"Duplicate edges: {result['duplicate_edges']}")
    print(f"Null values: {result['null_values']}")
    if 'negative_weights' in result:
        print(f"Negative weights: {result['negative_weights']}")


def demo_summary_statistics():
    """Demonstrate summary statistics generation."""
    print("\n" + "="*60)
    print("DEMO 4: Summary Statistics")
    print("="*60)

    # Create a more complex graph
    edges = pd.DataFrame({
        "source": ["A", "A", "B", "B", "C", "D", "E"],
        "target": ["B", "C", "C", "D", "D", "E", "A"],
        "amount": [100, 150, 200, 50, 75, 300, 125]
    })

    stats = graph_validation.graph_summary_statistics(
        edges,
        weight_col="amount"
    )

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Average Degree: {stats['avg_degree']:.2f}")
    print(f"\nDegree Distribution:")
    print(f"  Min: {stats['degree_stats']['min']}")
    print(f"  Max: {stats['degree_stats']['max']}")
    print(f"  Median: {stats['degree_stats']['median']:.2f}")
    print(f"  Std Dev: {stats['degree_stats']['std']:.2f}")
    print(f"\nWeight Statistics:")
    print(f"  Total: {stats['weight_stats']['sum']:.2f}")
    print(f"  Mean: {stats['weight_stats']['mean']:.2f}")
    print(f"  Range: [{stats['weight_stats']['min']:.2f}, {stats['weight_stats']['max']:.2f}]")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ASTROML GRAPH VALIDATION DEMO")
    print("="*60)

    demo_basic_validation()
    demo_isolated_nodes()
    demo_edge_consistency()
    demo_summary_statistics()

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60 + "\n")
