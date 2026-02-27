"""Tests for transaction graph module."""
import pytest
from astroml.features.transaction_graph import TransactionGraph


def test_basic_transaction():
    """Test adding a basic transaction."""
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0)
    
    assert "Alice" in graph.nodes
    assert "Bob" in graph.nodes
    assert graph.get_edge_weight("Alice", "Bob") == 100.0


def test_multi_asset_support():
    """Test multi-asset transaction support."""
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0, asset="USD")
    graph.add_transaction("Alice", "Bob", 0.5, asset="BTC")
    graph.add_transaction("Bob", "Charlie", 50.0, asset="USD")
    
    assert set(graph.get_assets()) == {"USD", "BTC"}
    assert graph.get_edge_weight("Alice", "Bob", asset="USD") == 100.0
    assert graph.get_edge_weight("Alice", "Bob", asset="BTC") == 0.5


def test_weighted_edges():
    """Test edge weight aggregation."""
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0)
    graph.add_transaction("Alice", "Bob", 50.0)
    graph.add_transaction("Alice", "Bob", 25.0)
    
    assert graph.get_edge_weight("Alice", "Bob", aggregation="sum") == 175.0
    assert graph.get_edge_weight("Alice", "Bob", aggregation="mean") == pytest.approx(58.33, rel=0.01)
    assert graph.get_edge_weight("Alice", "Bob", aggregation="count") == 3.0
    assert graph.get_edge_weight("Alice", "Bob", aggregation="max") == 100.0
    assert graph.get_edge_weight("Alice", "Bob", aggregation="min") == 25.0


def test_transaction_metadata():
    """Test transaction metadata storage."""
    graph = TransactionGraph()
    metadata = {"timestamp": "2024-01-01", "type": "payment"}
    graph.add_transaction("Alice", "Bob", 100.0, metadata=metadata)
    
    transactions = graph.get_transactions(from_account="Alice", to_account="Bob")
    assert len(transactions) == 1
    assert transactions[0]["metadata"] == metadata


def test_get_transactions_filtering():
    """Test transaction filtering."""
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0, asset="USD")
    graph.add_transaction("Alice", "Charlie", 50.0, asset="USD")
    graph.add_transaction("Bob", "Charlie", 25.0, asset="BTC")
    
    alice_txns = graph.get_transactions(from_account="Alice")
    assert len(alice_txns) == 2
    
    usd_txns = graph.get_transactions(asset="USD")
    assert len(usd_txns) == 2
    
    bob_to_charlie = graph.get_transactions(from_account="Bob", to_account="Charlie")
    assert len(bob_to_charlie) == 1
    assert bob_to_charlie[0]["asset"] == "BTC"


def test_networkx_export():
    """Test export to NetworkX format."""
    pytest.importorskip("networkx")
    
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0, asset="USD")
    graph.add_transaction("Alice", "Bob", 50.0, asset="USD")
    graph.add_transaction("Bob", "Charlie", 75.0, asset="USD")
    
    nx_graph = graph.to_networkx(aggregation="sum")
    
    assert len(nx_graph.nodes) == 3
    assert len(nx_graph.edges) == 2
    assert nx_graph["Alice"]["Bob"]["weight"] == 150.0
    assert nx_graph["Bob"]["Charlie"]["weight"] == 75.0


def test_networkx_export_with_metadata():
    """Test NetworkX export with metadata."""
    pytest.importorskip("networkx")
    
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0)
    graph.add_transaction("Alice", "Bob", 50.0)
    
    nx_graph = graph.to_networkx(include_metadata=True)
    
    assert nx_graph["Alice"]["Bob"]["transaction_count"] == 2
    assert "transactions" in nx_graph["Alice"]["Bob"]


def test_graph_summary():
    """Test graph summary statistics."""
    graph = TransactionGraph()
    graph.add_transaction("Alice", "Bob", 100.0, asset="USD")
    graph.add_transaction("Alice", "Charlie", 50.0, asset="USD")
    graph.add_transaction("Bob", "Charlie", 0.5, asset="BTC")
    
    summary = graph.summary()
    
    assert summary["node_count"] == 3
    assert summary["edge_count"] == 3
    assert summary["transaction_count"] == 3
    assert summary["asset_count"] == 2
    assert summary["assets"]["USD"] == 2
    assert summary["assets"]["BTC"] == 1


def test_empty_graph():
    """Test empty graph behavior."""
    graph = TransactionGraph()
    
    assert len(graph.nodes) == 0
    assert graph.get_edge_weight("Alice", "Bob") == 0.0
    assert graph.get_assets() == []
    assert graph.get_transactions() == []
