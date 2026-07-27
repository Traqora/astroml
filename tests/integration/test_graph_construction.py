"""Integration tests for graph construction and snapshot pipeline.

These tests verify the complete workflow from database operations
to graph construction, snapshot creation, and graph analysis.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from sqlalchemy.orm import Session

from astroml.db.schema import Operation, NormalizedTransaction
from astroml.features.graph.snapshot import (
    Edge,
    window_snapshot,
    snapshot_last_n_days,
    SnapshotWindow,
    iter_db_snapshots,
)
from astroml.features.transaction_graph import TransactionGraph


class TestGraphConstructionIntegration:
    """Integration tests for graph construction from database."""

    def test_build_graph_from_database_operations(
        self,
        populated_test_db: Session,
    ) -> None:
        """Test building a transaction graph from database operations."""
        # Query operations from database
        operations = populated_test_db.query(Operation).all()
        
        # Build graph
        graph = TransactionGraph()
        for op in operations:
            if op.destination_account:
                graph.add_transaction(
                    from_account=op.source_account,
                    to_account=op.destination_account,
                    amount=float(op.amount) if op.amount else 0.0,
                    asset=op.asset_code or "XLM",
                    metadata={"operation_type": op.type},
                )
        
        # Verify graph structure
        assert len(graph.nodes) > 0
        summary = graph.summary()
        assert summary["node_count"] > 0
        assert summary["transaction_count"] > 0

    def test_graph_with_multiple_assets(
        self,
    ) -> None:
        """Test graph construction with multiple asset types."""
        graph = TransactionGraph()
        
        # Add transactions with different assets
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("B", "C", 50.0, "USDC")
        graph.add_transaction("C", "A", 25.0, "BTC")
        graph.add_transaction("A", "C", 75.0, "XLM")
        
        # Verify multiple assets
        assets = graph.get_assets()
        assert len(assets) == 3
        assert "XLM" in assets
        assert "USDC" in assets
        assert "BTC" in assets

    def test_graph_edge_aggregation(
        self,
    ) -> None:
        """Test edge weight aggregation methods."""
        graph = TransactionGraph()
        
        # Add multiple transactions between same accounts
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("A", "B", 50.0, "XLM")
        graph.add_transaction("A", "B", 25.0, "XLM")
        
        # Test different aggregations
        sum_weight = graph.get_edge_weight("A", "B", aggregation="sum")
        mean_weight = graph.get_edge_weight("A", "B", aggregation="mean")
        count_weight = graph.get_edge_weight("A", "B", aggregation="count")
        max_weight = graph.get_edge_weight("A", "B", aggregation="max")
        min_weight = graph.get_edge_weight("A", "B", aggregation="min")
        
        assert sum_weight == 175.0
        assert mean_weight == 175.0 / 3
        assert count_weight == 3.0
        assert max_weight == 100.0
        assert min_weight == 25.0

    def test_graph_to_networkx_export(
        self,
    ) -> None:
        """Test exporting graph to NetworkX format."""
        graph = TransactionGraph()
        
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("B", "C", 50.0, "USDC")
        graph.add_transaction("C", "A", 25.0, "XLM")
        
        # Export to NetworkX
        nx_graph = graph.to_networkx()
        
        # Verify structure
        assert nx_graph.number_of_nodes() == 3
        assert nx_graph.number_of_edges() == 3
        
        # Verify edge weights
        assert nx_graph["A"]["B"]["weight"] == 100.0
        assert nx_graph["B"]["C"]["weight"] == 50.0

    def test_graph_summary_statistics(
        self,
    ) -> None:
        """Test graph summary statistics computation."""
        graph = TransactionGraph()
        
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("B", "C", 50.0, "USDC")
        graph.add_transaction("A", "C", 25.0, "XLM")
        graph.add_transaction("C", "A", 75.0, "BTC")
        
        summary = graph.summary()
        
        assert summary["node_count"] == 3
        assert summary["edge_count"] == 4
        assert summary["transaction_count"] == 4
        assert summary["asset_count"] == 3
        assert "XLM" in summary["assets"]
        assert summary["assets"]["XLM"] == 2


class TestGraphSnapshotIntegration:
    """Integration tests for graph snapshot creation."""

    def test_window_snapshot_creation(
        self,
    ) -> None:
        """Test creating a time-windowed graph snapshot."""
        base_time = int(datetime(2024, 1, 1).timestamp())
        
        edges = [
            Edge(src="A", dst="B", timestamp=base_time),
            Edge(src="B", dst="C", timestamp=base_time + 3600),  # +1 hour
            Edge(src="C", dst="D", timestamp=base_time + 7200),  # +2 hours
            Edge(src="D", dst="E", timestamp=base_time + 86400),  # +1 day
        ]
        
        # Create 12-hour window
        start_ts = base_time
        end_ts = base_time + 12 * 3600
        
        nodes, window_edges = window_snapshot(edges, start_ts, end_ts)
        
        # Should include first 3 edges (within 12 hours)
        assert len(window_edges) == 3
        assert len(nodes) == 4  # A, B, C, D
        assert "E" not in nodes

    def test_snapshot_last_n_days(
        self,
    ) -> None:
        """Test snapshot creation for last N days."""
        now_ts = int(datetime(2024, 1, 15).timestamp())
        
        edges = [
            Edge(src="A", dst="B", timestamp=now_ts - 86400),  # 1 day ago
            Edge(src="B", dst="C", timestamp=now_ts - 172800),  # 2 days ago
            Edge(src="C", dst="D", timestamp=now_ts - 259200),  # 3 days ago
            Edge(src="D", dst="E", timestamp=now_ts - 432000),  # 5 days ago
        ]
        
        # Get last 3 days
        nodes, window_edges = snapshot_last_n_days(edges, now_ts, days=3)
        
        # Should include edges from last 3 days
        assert len(window_edges) == 3
        assert len(nodes) == 4

    def test_snapshot_with_presorted_edges(
        self,
    ) -> None:
        """Test snapshot creation with pre-sorted edges."""
        base_time = int(datetime(2024, 1, 1).timestamp())
        
        edges = [
            Edge(src="A", dst="B", timestamp=base_time),
            Edge(src="B", dst="C", timestamp=base_time + 3600),
            Edge(src="C", dst="D", timestamp=base_time + 7200),
        ]
        
        # With presorted=True (should be faster)
        nodes1, edges1 = window_snapshot(edges, base_time, base_time + 7200, presorted=True)
        
        # With presorted=False (should sort first)
        nodes2, edges2 = window_snapshot(edges, base_time, base_time + 7200, presorted=False)
        
        # Results should be identical
        assert len(nodes1) == len(nodes2)
        assert len(edges1) == len(edges2)

    def test_empty_snapshot_window(
        self,
    ) -> None:
        """Test snapshot creation when no edges fall in window."""
        base_time = int(datetime(2024, 1, 1).timestamp())
        
        edges = [
            Edge(src="A", dst="B", timestamp=base_time),
            Edge(src="B", dst="C", timestamp=base_time + 3600),
        ]
        
        # Window with no edges
        nodes, window_edges = window_snapshot(
            edges, base_time + 7200, base_time + 10800
        )
        
        # Should be empty
        assert len(nodes) == 0
        assert len(window_edges) == 0


class TestDatabaseSnapshotIntegration:
    """Integration tests for database-backed snapshot creation."""

    def test_db_snapshot_from_normalized_transactions(
        self,
        test_session: Session,
    ) -> None:
        """Test creating snapshots from normalized transactions in database."""
        # Add normalized transactions
        base_time = datetime(2024, 1, 1)
        
        transactions = [
            NormalizedTransaction(
                transaction_hash="tx1",
                sender="G" + "A" * 55,
                receiver="G" + "B" * 55,
                asset="XLM",
                amount=100.0,
                timestamp=base_time,
            ),
            NormalizedTransaction(
                transaction_hash="tx2",
                sender="G" + "B" * 55,
                receiver="G" + "C" * 55,
                asset="USDC",
                amount=50.0,
                timestamp=base_time + timedelta(hours=1),
            ),
            NormalizedTransaction(
                transaction_hash="tx3",
                sender="G" + "C" * 55,
                receiver="G" + "A" * 55,
                asset="XLM",
                amount=25.0,
                timestamp=base_time + timedelta(hours=2),
            ),
        ]
        
        for tx in transactions:
            test_session.add(tx)
        test_session.commit()
        
        # Create snapshot
        t0 = base_time
        t_now = base_time + timedelta(hours=3)
        
        snapshots = list(iter_db_snapshots(
            window="1h",
            t0=t0,
            t_now=t_now,
            session=test_session,
        ))
        
        # Should have 3 hourly snapshots
        assert len(snapshots) == 3
        
        # Verify snapshot structure
        for snapshot in snapshots:
            assert isinstance(snapshot, SnapshotWindow)
            assert isinstance(snapshot.index, int)
            assert isinstance(snapshot.start, datetime)
            assert isinstance(snapshot.end, datetime)
            assert isinstance(snapshot.edges, list)
            assert isinstance(snapshot.nodes, set)

    def test_db_snapshot_with_rolling_window(
        self,
        test_session: Session,
    ) -> None:
        """Test creating rolling window snapshots from database."""
        base_time = datetime(2024, 1, 1)
        
        # Add transactions
        for i in range(10):
            tx = NormalizedTransaction(
                transaction_hash=f"tx{i}",
                sender=f"G{'A' * i}{'B' * (55-i)}",
                receiver=f"G{'C' * i}{'D' * (55-i)}",
                asset="XLM",
                amount=10.0 * i,
                timestamp=base_time + timedelta(hours=i),
            )
            test_session.add(tx)
        test_session.commit()
        
        # Create rolling snapshots (2-hour window, 1-hour step)
        t0 = base_time
        t_now = base_time + timedelta(hours=10)
        
        snapshots = list(iter_db_snapshots(
            window="2h",
            step="1h",
            t0=t0,
            t_now=t_now,
            session=test_session,
        ))
        
        # Should have 10 snapshots (rolling with overlap)
        assert len(snapshots) == 10


class TestGraphConstructionPipelineIntegration:
    """Integration tests for complete graph construction pipeline."""

    def test_database_to_graph_to_snapshot_pipeline(
        self,
        populated_test_db: Session,
    ) -> None:
        """Test complete pipeline from database to graph snapshot."""
        # Step 1: Extract operations from database
        operations = populated_test_db.query(Operation).all()
        
        # Step 2: Build transaction graph
        graph = TransactionGraph()
        for op in operations:
            if op.destination_account:
                graph.add_transaction(
                    from_account=op.source_account,
                    to_account=op.destination_account,
                    amount=float(op.amount) if op.amount else 0.0,
                    asset=op.asset_code or "XLM",
                )
        
        # Step 3: Convert to edge format for snapshot
        base_time = int(datetime(2024, 1, 1).timestamp())
        edges = []
        for src, dsts in graph.edges.items():
            for dst in dsts:
                for txn in graph.edges[src][dst]:
                    edges.append(Edge(src=src, dst=dst, timestamp=base_time))
        
        # Step 4: Create snapshot
        nodes, window_edges = window_snapshot(edges, base_time, base_time + 86400)
        
        # Verify pipeline
        assert len(graph.nodes) > 0
        assert len(edges) > 0
        assert len(nodes) > 0

    def test_incremental_graph_construction(
        self,
        test_session: Session,
    ) -> None:
        """Test incremental graph construction as new data arrives."""
        # Initial graph
        graph = TransactionGraph()
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("B", "C", 50.0, "USDC")
        
        initial_summary = graph.summary()
        assert initial_summary["transaction_count"] == 2
        
        # Add new transactions
        graph.add_transaction("C", "D", 25.0, "BTC")
        graph.add_transaction("D", "A", 75.0, "XLM")
        
        updated_summary = graph.summary()
        assert updated_summary["transaction_count"] == 4
        assert updated_summary["node_count"] == 4

    def test_graph_filtering_by_asset(
        self,
    ) -> None:
        """Test filtering graph by specific asset."""
        graph = TransactionGraph()
        
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("B", "C", 50.0, "USDC")
        graph.add_transaction("C", "A", 25.0, "XLM")
        graph.add_transaction("A", "D", 75.0, "BTC")
        
        # Filter by XLM
        xlm_txns = graph.get_transactions(asset="XLM")
        assert len(xlm_txns) == 2
        
        # Filter by USDC
        usdc_txns = graph.get_transactions(asset="USDC")
        assert len(usdc_txns) == 1

    def test_graph_persistence_workflow(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test saving and loading graph data."""
        graph = TransactionGraph()
        
        graph.add_transaction("A", "B", 100.0, "XLM")
        graph.add_transaction("B", "C", 50.0, "USDC")
        
        # Save graph summary
        summary = graph.summary()
        import json
        summary_path = temp_output_dir / "graph_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
        
        # Verify file exists
        assert summary_path.exists()
        
        # Load and verify
        with open(summary_path, 'r') as f:
            loaded_summary = json.load(f)
        
        assert loaded_summary["node_count"] == 3
        assert loaded_summary["transaction_count"] == 2
