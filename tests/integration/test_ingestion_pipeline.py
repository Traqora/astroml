"""End-to-end integration tests for the ingestion pipeline.

These tests verify the complete workflow from fetching ledger data
to storing it in the database, including parsing and state management.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest
from sqlalchemy.orm import Session

from astroml.db.schema import Ledger, Transaction, Operation, Account, Asset, Effect
from astroml.ingestion.service import IngestionService, IngestionResult
from astroml.ingestion.parsers import (
    parse_ledger,
    parse_transaction,
    parse_operation,
    parse_effect,
)
from astroml.ingestion.synthetic_fraud_injector import (
    inject_synthetic_fraud,
    SybilConfig,
    WashLoopConfig,
    InjectionSummary,
    run_injection,
)


class TestIngestionServiceIntegration:
    """Integration tests for IngestionService with database persistence."""

    def test_ingest_ledgers_to_database(
        self,
        test_session: Session,
        sample_ledger_data: List[Dict[str, Any]],
    ) -> None:
        """Test complete ingestion workflow from ledger data to database."""
        service = IngestionService()

        # Mock fetch function that returns ledger data
        def fetch_ledger(ledger_id: int) -> Dict[str, Any]:
            return sample_ledger_data[ledger_id - 1000]

        # Mock process function that stores in database
        def process_ledger(ledger_id: int, payload: Dict[str, Any]) -> None:
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        # Ingest ledgers
        result = service.ingest(
            start_ledger=1000,
            end_ledger=1001,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
        )

        # Verify results
        assert result.attempted == [1000, 1001]
        assert result.processed == [1000, 1001]
        assert result.skipped == []

        # Verify database state
        ledgers = test_session.query(Ledger).all()
        assert len(ledgers) == 2
        assert ledgers[0].sequence == 1000
        assert ledgers[1].sequence == 1001

    def test_ingest_with_idempotency(
        self,
        test_session: Session,
        sample_ledger_data: List[Dict[str, Any]],
    ) -> None:
        """Test that ingestion is idempotent - re-processing skips already processed ledgers."""
        service = IngestionService()

        def fetch_ledger(ledger_id: int) -> Dict[str, Any]:
            return sample_ledger_data[ledger_id - 1000]

        def process_ledger(ledger_id: int, payload: Dict[str, Any]) -> None:
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        # First ingestion
        result1 = service.ingest(
            start_ledger=1000,
            end_ledger=1001,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
        )
        assert result1.processed == [1000, 1001]

        # Second ingestion - should skip already processed
        result2 = service.ingest(
            start_ledger=1000,
            end_ledger=1001,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
        )
        assert result2.attempted == [1000, 1001]
        assert result2.processed == []
        assert result2.skipped == [1000, 1001]

        # Verify no duplicates in database
        ledgers = test_session.query(Ledger).all()
        assert len(ledgers) == 2

    def test_ingest_with_partial_failure(
        self,
        test_session: Session,
        sample_ledger_data: List[Dict[str, Any]],
    ) -> None:
        """Test ingestion continues even if one ledger fails to process."""
        service = IngestionService()

        def fetch_ledger(ledger_id: int) -> Dict[str, Any]:
            return sample_ledger_data[ledger_id - 1000]

        call_count = [0]

        def process_ledger(ledger_id: int, payload: Dict[str, Any]) -> None:
            call_count[0] += 1
            if ledger_id == 1000:
                raise ValueError("Simulated failure")
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        # Should fail on first ledger
        with pytest.raises(ValueError):
            service.ingest(
                start_ledger=1000,
                end_ledger=1001,
                fetch_fn=fetch_ledger,
                process_fn=process_ledger,
            )

        # State should not have marked ledger 1000 as processed
        # Retry without the failing ledger
        result = service.ingest(
            start_ledger=1001,
            end_ledger=1001,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
        )
        assert result.processed == [1001]

        # Verify only successful ledger is in database
        ledgers = test_session.query(Ledger).all()
        assert len(ledgers) == 1
        assert ledgers[0].sequence == 1001


class TestParserIntegration:
    """Integration tests for parsers with database storage."""

    def test_parse_and_store_complete_transaction(
        self,
        test_session: Session,
        sample_transaction_data: List[Dict[str, Any]],
        sample_operation_data: List[Dict[str, Any]],
    ) -> None:
        """Test parsing and storing a complete transaction with operations."""
        # First, add a ledger
        ledger = Ledger(
            sequence=1000,
            hash="a" * 64,
            closed_at=datetime(2024, 1, 1),
            successful_transaction_count=1,
            failed_transaction_count=0,
            operation_count=2,
        )
        test_session.add(ledger)
        test_session.commit()

        # Parse and store transaction
        tx_data = sample_transaction_data[0]
        transaction = parse_transaction(tx_data)
        test_session.add(transaction)
        test_session.commit()

        # Parse and store operations
        for i, op_data in enumerate(sample_operation_data):
            if op_data["transaction_hash"] == tx_data["hash"]:
                operation = parse_operation(op_data, application_order=i)
                test_session.add(operation)
        test_session.commit()

        # Verify transaction was stored
        stored_tx = test_session.query(Transaction).filter_by(hash=tx_data["hash"]).first()
        assert stored_tx is not None
        assert stored_tx.source_account == tx_data["source_account"]
        assert stored_tx.ledger_sequence == 1000

        # Verify operations were stored and linked
        operations = test_session.query(Operation).filter_by(transaction_hash=tx_data["hash"]).all()
        assert len(operations) == 2

    def test_parse_and_store_effects(
        self,
        test_session: Session,
        sample_effect_data: List[Dict[str, Any]],
    ) -> None:
        """Test parsing and storing effects."""
        for effect_data in sample_effect_data:
            effect = parse_effect(effect_data)
            test_session.add(effect)
        test_session.commit()

        # Verify effects were stored
        effects = test_session.query(Effect).all()
        assert len(effects) == 2
        assert effects[0].type == "account_debited"
        assert effects[1].type == "account_credited"


class TestSyntheticFraudInjectionIntegration:
    """Integration tests for synthetic fraud injection."""

    def test_inject_fraud_patterns_to_file(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test injecting fraud patterns and saving to file."""
        # Create sample clean ledger
        clean_ledger = [
            {
                "source_account": "G" + "A" * 55,
                "destination_account": "G" + "B" * 55,
                "amount": 100.0,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        input_file = temp_data_dir / "clean_ledger.jsonl"
        output_file = temp_data_dir / "augmented_ledger.jsonl"
        summary_file = temp_data_dir / "summary.json"

        # Write clean ledger
        with open(input_file, "w") as f:
            for tx in clean_ledger:
                f.write(tx.__str__() + "\n")

        # Run injection
        summary = run_injection(
            input_path=str(input_file),
            output_path=str(output_file),
            summary_path=str(summary_file),
            seed=42,
            sybil=SybilConfig(clusters=1, cluster_size=3, tx_per_member=2),
            wash=WashLoopConfig(loops=1, loop_size=3, rounds=2),
            source_field="source_account",
            dest_field="destination_account",
            amount_field="amount",
            timestamp_field="created_at",
        )

        # Verify summary
        assert summary.original_transactions == 1
        assert summary.sybil_transactions == 6  # 1 cluster * 3 members * 2 tx
        assert summary.wash_loop_transactions == 6  # 1 loop * 3 accounts * 2 rounds
        assert summary.injected_transactions == 12
        assert summary.total_transactions == 13

        # Verify output file exists
        assert output_file.exists()
        assert summary_file.exists()

    def test_inject_fraud_in_memory(
        self,
    ) -> None:
        """Test injecting fraud patterns in memory."""
        clean_transactions = [
            {
                "source_account": "G" + "A" * 55,
                "destination_account": "G" + "B" * 55,
                "amount": 100.0,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        augmented, summary = inject_synthetic_fraud(
            clean_transactions,
            seed=42,
            sybil=SybilConfig(clusters=1, cluster_size=2, tx_per_member=1),
            wash=WashLoopConfig(loops=0, loop_size=0, rounds=0),  # No wash loops
            source_field="source_account",
            dest_field="destination_account",
            amount_field="amount",
            timestamp_field="created_at",
        )

        # Verify augmentation
        assert len(augmented) == 3  # 1 original + 2 sybil transactions
        assert summary.original_transactions == 1
        assert summary.sybil_transactions == 2
        assert summary.wash_loop_transactions == 0

        # Verify synthetic transactions are tagged
        synthetic_txs = [tx for tx in augmented if tx.get("synthetic_fraud")]
        assert len(synthetic_txs) == 2
        assert all(tx["fraud_pattern"] == "sybil_cluster" for tx in synthetic_txs)

    def test_fraud_injection_preserves_original_data(
        self,
    ) -> None:
        """Test that fraud injection preserves original transaction data."""
        original = [
            {
                "source_account": "G" + "A" * 55,
                "destination_account": "G" + "B" * 55,
                "amount": 100.0,
                "created_at": "2024-01-01T00:00:00Z",
                "custom_field": "should_preserve",
            }
        ]

        augmented, _ = inject_synthetic_fraud(
            original,
            seed=42,
            sybil=SybilConfig(clusters=0, cluster_size=0, tx_per_member=0),
            wash=WashLoopConfig(loops=0, loop_size=0, rounds=0),
        )

        # Original transaction should be unchanged
        assert len(augmented) == 1
        assert augmented[0]["custom_field"] == "should_preserve"
        assert "synthetic_fraud" not in augmented[0]


class TestCompleteIngestionWorkflow:
    """Integration tests for the complete ingestion workflow."""

    def test_ledger_to_operations_workflow(
        self,
        test_session: Session,
    ) -> None:
        """Test complete workflow from ledger to operations."""
        # Create ledger
        ledger_data = {
            "sequence": 1000,
            "hash": "a" * 64,
            "prev_hash": "b" * 64,
            "closed_at": datetime(2024, 1, 1),
            "successful_transaction_count": 1,
            "failed_transaction_count": 0,
            "operation_count": 2,
        }
        ledger = Ledger(**ledger_data)
        test_session.add(ledger)
        test_session.commit()

        # Create transaction
        tx_data = {
            "hash": "tx1" + "a" * 60,
            "ledger": 1000,
            "source_account": "G" + "A" * 55,
            "created_at": datetime(2024, 1, 1),
            "fee_charged": 100,
            "operation_count": 2,
            "successful": True,
            "memo_type": "none",
        }
        transaction = parse_transaction(tx_data)
        test_session.add(transaction)
        test_session.commit()

        # Create operations
        op_data_1 = {
            "id": 1,
            "transaction_hash": "tx1" + "a" * 60,
            "source_account": "G" + "A" * 55,
            "type": "payment",
            "to": "G" + "B" * 55,
            "amount": "100.0",
            "asset_type": "native",
            "created_at": datetime(2024, 1, 1),
        }
        op_data_2 = {
            "id": 2,
            "transaction_hash": "tx1" + "a" * 60,
            "source_account": "G" + "A" * 55,
            "type": "create_account",
            "account": "G" + "C" * 55,
            "starting_balance": "50.0",
            "created_at": datetime(2024, 1, 1),
        }

        op1 = parse_operation(op_data_1, application_order=0)
        op2 = parse_operation(op_data_2, application_order=1)
        test_session.add(op1)
        test_session.add(op2)
        test_session.commit()

        # Verify complete chain
        assert test_session.query(Ledger).count() == 1
        assert test_session.query(Transaction).count() == 1
        assert test_session.query(Operation).count() == 2

        # Verify relationships
        stored_tx = test_session.query(Transaction).first()
        assert stored_tx.ledger_sequence == 1000
        assert len(stored_tx.operations) == 2

    def test_incremental_ingestion_with_state(
        self,
        test_session: Session,
        sample_ledger_data: List[Dict[str, Any]],
    ) -> None:
        """Test incremental ingestion with state persistence."""
        service = IngestionService()

        def fetch_ledger(ledger_id: int) -> Dict[str, Any]:
            return sample_ledger_data[ledger_id - 1000]

        def process_ledger(ledger_id: int, payload: Dict[str, Any]) -> None:
            ledger = parse_ledger(payload)
            test_session.add(ledger)
            test_session.commit()

        # First batch
        result1 = service.ingest(
            start_ledger=1000,
            end_ledger=1000,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
        )
        assert result1.processed == [1000]

        # Second batch - should continue from where we left off
        result2 = service.ingest(
            start_ledger=1001,
            end_ledger=1001,
            fetch_fn=fetch_ledger,
            process_fn=process_ledger,
        )
        assert result2.processed == [1001]

        # Verify both ledgers are in database
        assert test_session.query(Ledger).count() == 2
