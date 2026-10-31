"""Integration tests for graph construction and snapshot pipeline.

These tests verify the complete workflow from database operations
to graph construction, snapshot creation, and graph analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from astroml.db.schema import NormalizedTransaction, Operation
from astroml.features.graph.snapshot import (
    Edge,
    SnapshotWindow,
    iter_db_snapshot_edges,
    iter_db_snapshots,
    parallel_build_snapshots,
    snapshot_last_n_days,
    window_snapshot,
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

        # Get last 3 days.  The list above runs newest-first, so it is not
        # pre-sorted by timestamp and must say so — issue #732 rejects a
        # presorted edge list that is not ordered rather than bisecting an
        # unsorted array and returning a window that is quietly wrong.
        nodes, window_edges = snapshot_last_n_days(edges, now_ts, days=3, presorted=False)

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
        nodes, window_edges = window_snapshot(edges, base_time + 7200, base_time + 10800)

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

        snapshots = list(
            iter_db_snapshots(
                window="1h",
                t0=t0,
                t_now=t_now,
                session=test_session,
            )
        )

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

        snapshots = list(
            iter_db_snapshots(
                window="2h",
                step="1h",
                t0=t0,
                t_now=t_now,
                session=test_session,
            )
        )

        # Should have 10 snapshots (rolling with overlap)
        assert len(snapshots) == 10


class TestSnapshotWindowBoundaries:
    """Regression tests for DB-backed snapshot window boundaries (issue #834).

    Two separate bugs hid behind every existing snapshot test here: none of
    them put a transaction at the exact instant a non-overlapping window
    boundary falls on, so neither was ever exercised.

    1. ``iter_db_snapshots``/``iter_db_snapshot_edges``/``_build_snapshot_window``
       all normalised ``t0``/``t_now`` to tz-*aware* UTC, then bound a
       tz-aware ``window_start``/``window_end`` straight into a
       ``WHERE timestamp >= :bound`` comparison against
       ``normalized_transactions.timestamp``, a naive ``DateTime`` column.
       psycopg converts a tz-aware bind to the *client's local timezone*
       before dropping the tzinfo to match the naive column, so on any host
       not running with ``TZ=UTC`` every window boundary silently shifted by
       the host's UTC offset — a transaction could land in the wrong hour's
       (or day's) window with nothing raising. Fixed by ``_naive_utc_bound``.

    2. ``window_end = min(window_start + win_delta, t_now)`` makes window N's
       end equal window N+1's start for non-overlapping windows. The query
       for each window compared both sides with ``<=``, so a row landing
       exactly on that shared instant matched both windows' filters and was
       double-counted. Fixed by ``_end_bound_filter``: the right bound is
       exclusive except on the very last window, where it stays inclusive
       of ``t_now`` so a row at exactly the end of the whole range isn't
       dropped.
    """

    def test_boundary_transaction_is_not_double_counted_across_adjacent_windows(
        self,
        test_session: Session,
    ) -> None:
        """A transaction exactly on a non-overlapping window boundary
        belongs to the later window only, not both."""
        base_time = datetime(2024, 1, 1)

        transactions = [
            NormalizedTransaction(
                transaction_hash="tx_before",
                sender="G" + "A" * 55,
                receiver="G" + "B" * 55,
                asset="XLM",
                amount=100.0,
                timestamp=base_time,
            ),
            NormalizedTransaction(
                transaction_hash="tx_on_boundary",
                sender="G" + "B" * 55,
                receiver="G" + "C" * 55,
                asset="XLM",
                amount=50.0,
                timestamp=base_time + timedelta(hours=1),
            ),
            NormalizedTransaction(
                transaction_hash="tx_at_end",
                sender="G" + "C" * 55,
                receiver="G" + "D" * 55,
                asset="XLM",
                amount=25.0,
                timestamp=base_time + timedelta(hours=3),
            ),
        ]
        for tx in transactions:
            test_session.add(tx)
        test_session.commit()

        t_now = base_time + timedelta(hours=3)
        snapshots = list(
            iter_db_snapshots(window="1h", t0=base_time, t_now=t_now, session=test_session)
        )

        total_edges = sum(len(s.edges) for s in snapshots)
        assert total_edges == 3, (
            f"expected each of the 3 transactions counted exactly once across "
            f"all windows, got {total_edges} total edges"
        )

        # window 0 = [base_time, base_time+1h): only tx_before
        assert [e.src for e in snapshots[0].edges] == [transactions[0].sender]
        # window 1 = [base_time+1h, base_time+2h): only tx_on_boundary, not
        # shared with window 0 or window 2
        assert [e.src for e in snapshots[1].edges] == [transactions[1].sender]
        # the final window's end equals t_now and stays inclusive, so the
        # transaction landing exactly on t_now is still counted there
        assert [e.src for e in snapshots[2].edges] == [transactions[2].sender]

    def test_streaming_variant_does_not_double_count_boundary_transaction(
        self,
        test_session: Session,
    ) -> None:
        """iter_db_snapshot_edges (the streaming variant) applies the same
        boundary fix as iter_db_snapshots."""
        base_time = datetime(2024, 1, 1)

        transactions = [
            NormalizedTransaction(
                transaction_hash="tx_before",
                sender="G" + "A" * 55,
                receiver="G" + "B" * 55,
                asset="XLM",
                amount=100.0,
                timestamp=base_time,
            ),
            NormalizedTransaction(
                transaction_hash="tx_on_boundary",
                sender="G" + "B" * 55,
                receiver="G" + "C" * 55,
                asset="XLM",
                amount=50.0,
                timestamp=base_time + timedelta(hours=1),
            ),
        ]
        for tx in transactions:
            test_session.add(tx)
        test_session.commit()

        t_now = base_time + timedelta(hours=2)
        total_edges = 0
        per_window_counts = []
        for _meta, edges_iter in iter_db_snapshot_edges(
            window="1h", t0=base_time, t_now=t_now, session=test_session
        ):
            edges = list(edges_iter)
            per_window_counts.append(len(edges))
            total_edges += len(edges)

        assert total_edges == 2, f"expected 2 total edges, got {total_edges}"
        assert per_window_counts == [1, 1]

    def test_parallel_build_snapshots_does_not_double_count_boundary_transaction(
        self,
        test_session: Session,
    ) -> None:
        """parallel_build_snapshots (the joblib-parallel builder) applies the
        same boundary fix as the sequential and streaming variants."""
        base_time = datetime(2024, 1, 1)

        transactions = [
            NormalizedTransaction(
                transaction_hash="tx_before",
                sender="G" + "A" * 55,
                receiver="G" + "B" * 55,
                asset="XLM",
                amount=100.0,
                timestamp=base_time,
            ),
            NormalizedTransaction(
                transaction_hash="tx_on_boundary",
                sender="G" + "B" * 55,
                receiver="G" + "C" * 55,
                asset="XLM",
                amount=50.0,
                timestamp=base_time + timedelta(hours=1),
            ),
        ]
        for tx in transactions:
            test_session.add(tx)
        test_session.commit()

        t_now = base_time + timedelta(hours=2)
        windows = parallel_build_snapshots(window="1h", t0=base_time, t_now=t_now, n_jobs=1)

        total_edges = sum(len(w.edges) for w in windows)
        assert total_edges == 2, f"expected 2 total edges, got {total_edges}"

    def test_rolling_windows_still_legitimately_overlap(
        self,
        test_session: Session,
    ) -> None:
        """The exclusive-end fix only removes the artefact between adjacent
        *non-overlapping* windows; rolling windows (step < window) are meant
        to share rows and must keep doing so."""
        base_time = datetime(2024, 1, 1)

        transactions = [
            NormalizedTransaction(
                transaction_hash=f"tx{i}",
                sender=f"G{'A' * i}{'B' * (55 - i)}",
                receiver=f"G{'C' * i}{'D' * (55 - i)}",
                asset="XLM",
                amount=10.0 * i,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(3)
        ]
        for tx in transactions:
            test_session.add(tx)
        test_session.commit()

        t_now = base_time + timedelta(hours=2)
        snapshots = list(
            iter_db_snapshots(
                window="2h", step="1h", t0=base_time, t_now=t_now, session=test_session
            )
        )

        # window 0 = [base_time, base_time+2h] (last window, inclusive): tx0, tx1, tx2
        # window 1 = [base_time+1h, base_time+2h] (last window, inclusive): tx1, tx2
        assert len(snapshots[0].edges) == 3
        assert len(snapshots[1].edges) == 2

    def test_boundary_fix_does_not_change_total_transaction_count_vs_naive_sum(
        self,
        test_session: Session,
    ) -> None:
        """Sanity check at a larger scale: for non-overlapping windows, the
        sum of each window's edge count must equal the total number of
        transactions in range, with no row dropped or duplicated, even when
        several fall exactly on window boundaries."""
        base_time = datetime(2024, 1, 1)

        # One transaction every 30 minutes for 5 hours: several land exactly
        # on a 1-hour window boundary (hour marks), others fall mid-window.
        transactions = [
            NormalizedTransaction(
                transaction_hash=f"tx{i}",
                sender=f"G{'A' * (i % 50 + 1)}{'B' * (55 - (i % 50 + 1))}",
                receiver=f"G{'C' * (i % 50 + 1)}{'D' * (55 - (i % 50 + 1))}",
                asset="XLM",
                amount=1.0,
                timestamp=base_time + timedelta(minutes=30 * i),
            )
            for i in range(10)
        ]
        for tx in transactions:
            test_session.add(tx)
        test_session.commit()

        t_now = base_time + timedelta(hours=4, minutes=30)
        snapshots = list(
            iter_db_snapshots(window="1h", t0=base_time, t_now=t_now, session=test_session)
        )

        total_edges = sum(len(s.edges) for s in snapshots)
        assert total_edges == len(transactions), (
            f"expected all {len(transactions)} transactions counted exactly "
            f"once, got {total_edges} total edges across {len(snapshots)} windows"
        )


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
        with open(summary_path, "w") as f:
            json.dump(summary, f)

        # Verify file exists
        assert summary_path.exists()

        # Load and verify
        with open(summary_path) as f:
            loaded_summary = json.load(f)

        assert loaded_summary["node_count"] == 3
        assert loaded_summary["transaction_count"] == 2
