"""Integration tests for the ingestion-to-features pipeline.

This module tests the complete data pipeline from ledger ingestion through
feature computation, validating data integrity at each step and testing
error recovery scenarios.

Issue: #513
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from astroml.db.schema import Account, Ledger, Operation, Transaction
from astroml.features.frequency import compute_daily_transaction_counts
from astroml.features.node_features import compute_node_features
from astroml.features.structural_importance import compute_degree_centrality
from astroml.features.transaction_graph import TransactionGraph
from astroml.ingestion.parsers import parse_ledger, parse_operation, parse_transaction
from astroml.ingestion.service import IngestionResult, IngestionService
from astroml.ingestion.state import StateStore


class TestIngestionToFeaturesPipeline:
    """Integration tests for ingestion-to-features pipeline."""

    def test_ledger_ingestion_to_feature_computation(
        self,
        test_session: Session,
        tmp_path: Path,
    ) -> None:
        """Test complete pipeline: ingest ledgers → build graph → compute features → validate results."""
        # Step 1: Ingest sample ledgers
        ledgers = []
        for i in range(5):
            ledger_data = {
                "sequence": 1000 + i,
                "hash": "a" * 64,
                "prev_hash": "b" * 64,
                "closed_at": datetime(2024, 1, 1) + timedelta(hours=i),
                "successful_transaction_count": 2,
                "failed_transaction_count": 0,
                "operation_count": 4,
            }
            ledger = parse_ledger(ledger_data)
            test_session.add(ledger)
            ledgers.append(ledger)

        test_session.commit()

        # Step 2: Ingest transactions for each ledger
        accounts = ["G" + chr(65 + i) * 55 for i in range(5)]  # GAAAA..., GBBBB...
        for i, ledger in enumerate(ledgers):
            for j in range(2):
                tx_data = {
                    "hash": f"tx{i}_{j}" + "a" * 60,
                    "ledger": ledger.sequence,
                    "source_account": accounts[i % len(accounts)],
                    "destination_account": accounts[(i + 1) % len(accounts)],
                    "created_at": ledger.closed_at,
                    "fee_charged": 100 + j * 50,
                    "operation_count": 2,
                    "successful": True,
                    "memo_type": "none",
                }
                tx = parse_transaction(tx_data)
                test_session.add(tx)

        test_session.commit()

        # Step 3: Ingest operations
        tx_count = test_session.query(Transaction).count()
        for i, tx in enumerate(test_session.query(Transaction).all()):
            op_data = {
                "id": i + 1,
                "transaction_hash": tx.hash,
                "source_account": tx.source_account,
                "type": "payment",
                "to": accounts[(i + 2) % len(accounts)],
                "amount": str(100.0 + i * 10),
                "asset_type": "native",
                "created_at": tx.created_at,
            }
            op = parse_operation(op_data, application_order=0)
            test_session.add(op)

        test_session.commit()

        # Step 4: Validate data integrity at each step
        # Verify ledger count
        ledger_count = test_session.query(Ledger).count()
        assert ledger_count == 5, f"Expected 5 ledgers, got {ledger_count}"

        # Verify transaction count
        transaction_count = test_session.query(Transaction).count()
        assert transaction_count == 10, f"Expected 10 transactions, got {transaction_count}"

        # Verify operation count
        operation_count = test_session.query(Operation).count()
        assert operation_count == 10, f"Expected 10 operations, got {operation_count}"

        # Step 5: Build graph from operations
        operations = test_session.query(Operation).all()
        edges = []
        for op in operations:
            if op.destination_account:
                edges.append(
                    {
                        "src": op.source_account,
                        "dst": op.destination_account,
                        "amount": float(op.amount) if op.amount else 0.0,
                        "timestamp": op.created_at.timestamp(),
                        "asset": op.asset_code or "XLM",
                    }
                )

        # Step 6: Compute features
        features_df = compute_node_features(edges)

        # Validate feature computation
        assert not features_df.empty, "Feature DataFrame should not be empty"
        assert len(features_df) > 0, "Should have features for at least one node"
        assert all(features_df.columns), "All feature columns should be non-empty"

        # Verify features are finite
        assert np.all(np.isfinite(features_df.values)), "All feature values should be finite"

        # Step 7: Test graph building
        graph = TransactionGraph()
        for edge in edges:
            graph.add_transaction(
                from_account=edge["src"],
                to_account=edge["dst"],
                amount=edge["amount"],
                asset=edge["asset"],
            )

        summary = graph.summary()
        assert summary["node_count"] > 0, "Graph should have nodes"
        assert summary["transaction_count"] == len(edges), "Transaction count should match edges"

        # Step 8: Test frequency features
        tx_df = pd.DataFrame(
            [
                {
                    "account_id": tx.source_account,
                    "timestamp": tx.created_at,
                }
                for tx in test_session.query(Transaction).all()
            ]
        )

        if not tx_df.empty:
            daily_counts = compute_daily_transaction_counts(tx_df, "account_id", "timestamp")
            assert not daily_counts.empty, "Daily counts should not be empty"
            assert np.all(np.isfinite(daily_counts.values)), "Daily counts should be finite"

    def test_ingestion_service_idempotency(
        self,
        test_session: Session,
        tmp_path: Path,
    ) -> None:
        """Test that ingestion service is idempotent - re-processing same ledgers skips them."""
        # Setup state store
        state_file = tmp_path / "ingestion_state.json"
        state_store = StateStore(state_path=str(state_file))
        service = IngestionService(state_store=state_store)

        # Create ledger data
        ledger_sequence = 1000
        processed_ledgers = []

        def fetch_ledger(ledger_id: int) -> Dict[str, Any]:
            return {
                "sequence": ledger_id,
                "hash": "a" * 64,
                "closed_at": datetime(2024, 1, 1),
                "successful_transaction_count": 1,
                "failed_transaction_count": 0,
                "operation_count": 1,
            }

        def process_ledger(ledger_id: int, payload: Dict[str, Any]) -> None:
            processed_ledgers.append(ledger_id)
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        # First ingestion
        result1 = service.ingest(
            start_ledger=ledger_sequence,
            end_ledger=ledger_sequence + 2,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
            batch_size=1,
        )

        assert len(result1.processed) == 3, "Should process 3 ledgers on first run"
        assert len(result1.skipped) == 0, "Should skip 0 ledgers on first run"

        # Clear processed list but keep state
        processed_ledgers.clear()

        # Second ingestion with same range
        result2 = service.ingest(
            start_ledger=ledger_sequence,
            end_ledger=ledger_sequence + 2,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
            batch_size=1,
        )

        assert len(result2.processed) == 0, "Should process 0 ledgers on second run (idempotent)"
        assert len(result2.skipped) == 3, "Should skip 3 ledgers on second run"
        assert len(processed_ledgers) == 0, "Process function should not be called"

    def test_ingestion_error_recovery_mid_batch(
        self,
        test_session: Session,
        tmp_path: Path,
    ) -> None:
        """Test error recovery when ingestion fails mid-batch."""
        state_file = tmp_path / "ingestion_state_recovery.json"
        state_store = StateStore(state_path=str(state_file))
        service = IngestionService(state_store=state_store)

        ledger_sequence = 2000
        processed_ledgers = []

        def fetch_ledger(ledger_id: int) -> Dict[str, Any]:
            return {
                "sequence": ledger_id,
                "hash": "a" * 64,
                "closed_at": datetime(2024, 1, 1),
                "successful_transaction_count": 1,
                "failed_transaction_count": 0,
                "operation_count": 1,
            }

        def process_ledger_with_failure(ledger_id: int, payload: Dict[str, Any]) -> None:
            processed_ledgers.append(ledger_id)
            # Fail on ledger 2002
            if ledger_id == 2002:
                raise ValueError("Simulated processing failure")
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        # First ingestion that fails mid-batch
        with pytest.raises(ValueError, match="Simulated processing failure"):
            service.ingest(
                start_ledger=ledger_sequence,
                end_ledger=ledger_sequence + 4,
                fetch_fn=fetch_ledger,
                process_fn=process_ledger_with_failure,
                batch_size=2,  # Batch size 2 means state is flushed every 2 ledgers
            )

        # Should have processed ledgers before failure
        assert 2000 in processed_ledgers, "Ledger 2000 should be processed"
        assert 2001 in processed_ledgers, "Ledger 2001 should be processed"
        assert 2002 in processed_ledgers, "Ledger 2002 should be attempted (and failed)"

        # State should have saved progress for batch
        state = state_store.load()
        # Depending on batch size and when failure occurred, some ledgers may be saved

        # Recovery: restart ingestion from last known good state
        processed_ledgers.clear()

        def process_ledger_success(ledger_id: int, payload: Dict[str, Any]) -> None:
            processed_ledgers.append(ledger_id)
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        result = service.ingest(
            start_ledger=ledger_sequence,
            end_ledger=ledger_sequence + 4,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger_success,
            batch_size=2,
        )

        # Should complete successfully
        assert len(result.processed) > 0, "Should process remaining ledgers"

        # Verify database state
        ledger_count = test_session.query(Ledger).count()
        assert ledger_count == 5, f"Should have 5 ledgers after recovery, got {ledger_count}"

    def test_feature_computation_with_empty_data(
        self,
        test_session: Session,
    ) -> None:
        """Test feature computation handles empty data gracefully."""
        # Empty operations
        edges = []

        features_df = compute_node_features(edges)

        # Should return empty DataFrame or handle gracefully
        assert (
            features_df.empty or len(features_df) == 0
        ), "Empty edges should produce empty features"

    def test_feature_computation_with_single_node(
        self,
        test_session: Session,
    ) -> None:
        """Test feature computation with single node (edge case)."""
        edges = [
            {
                "src": "GAAAA",
                "dst": "GBBBB",
                "amount": 100.0,
                "timestamp": 1000,
                "asset": "XLM",
            }
        ]

        features_df = compute_node_features(edges)

        # Should compute features for both nodes
        assert len(features_df) == 2, "Should have features for 2 nodes"
        assert np.all(np.isfinite(features_df.values)), "All features should be finite"

    def test_graph_building_consistency(
        self,
        test_session: Session,
    ) -> None:
        """Test graph building produces consistent results."""
        edges = [
            {"src": "A", "dst": "B", "amount": 100.0, "timestamp": 1000, "asset": "XLM"},
            {"src": "B", "dst": "C", "amount": 50.0, "timestamp": 1001, "asset": "XLM"},
            {"src": "C", "dst": "A", "amount": 75.0, "timestamp": 1002, "asset": "XLM"},
        ]

        # Build graph twice
        graph1 = TransactionGraph()
        for edge in edges:
            graph1.add_transaction(
                from_account=edge["src"],
                to_account=edge["dst"],
                amount=edge["amount"],
                asset=edge["asset"],
            )

        graph2 = TransactionGraph()
        for edge in edges:
            graph2.add_transaction(
                from_account=edge["src"],
                to_account=edge["dst"],
                amount=edge["amount"],
                asset=edge["asset"],
            )

        summary1 = graph1.summary()
        summary2 = graph2.summary()

        assert summary1["node_count"] == summary2["node_count"], "Node count should be consistent"
        assert (
            summary1["transaction_count"] == summary2["transaction_count"]
        ), "Transaction count should be consistent"

    def test_centrality_features_computation(
        self,
        test_session: Session,
    ) -> None:
        """Test centrality feature computation from database operations."""
        # Create test operations
        accounts = ["G" + chr(65 + i) * 55 for i in range(4)]
        base_time = datetime(2024, 1, 1)

        for i in range(6):
            ledger = Ledger(
                sequence=1000 + i,
                hash="a" * 64,
                closed_at=base_time + timedelta(hours=i),
                successful_transaction_count=1,
                failed_transaction_count=0,
                operation_count=1,
            )
            test_session.add(ledger)

            tx = Transaction(
                hash=f"tx{i}" + "a" * 60,
                ledger_sequence=1000 + i,
                source_account=accounts[i % len(accounts)],
                destination_account=accounts[(i + 1) % len(accounts)],
                created_at=base_time + timedelta(hours=i),
                fee=100,
                operation_count=1,
                successful=True,
                memo_type="none",
            )
            test_session.add(tx)

            op = Operation(
                id=i + 1,
                transaction_hash=tx.hash,
                source_account=tx.source_account,
                type="payment",
                destination_account=tx.destination_account,
                amount=str(100.0 + i * 10),
                asset_code="XLM",
                asset_type="native",
                created_at=tx.created_at,
            )
            test_session.add(op)

        test_session.commit()

        # Extract edges
        operations = test_session.query(Operation).all()
        edges = [
            {
                "src": op.source_account,
                "dst": op.destination_account,
                "amount": float(op.amount) if op.amount else 0.0,
                "timestamp": op.created_at.timestamp(),
                "asset": op.asset_code or "XLM",
            }
            for op in operations
            if op.destination_account
        ]

        # Compute centrality features
        degree_centrality = compute_degree_centrality(edges, weighted=False)

        # Validate centrality
        assert not degree_centrality.empty, "Centrality should not be empty"
        assert np.all(degree_centrality.values >= 0), "Centrality values should be non-negative"
        assert np.all(degree_centrality.values <= 1), "Normalized centrality should be <= 1"
        assert np.all(np.isfinite(degree_centrality.values)), "Centrality values should be finite"
