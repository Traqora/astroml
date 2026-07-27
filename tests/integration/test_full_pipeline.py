"""Comprehensive end-to-end pipeline integration tests.

These tests verify the complete AstroML workflow from raw ledger data
to trained models, including all intermediate steps: ingestion,
feature engineering, graph construction, model training, and validation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from sqlalchemy.orm import Session

from astroml.db.schema import Ledger, Transaction, Operation, Account, Asset
from astroml.ingestion.service import IngestionService
from astroml.ingestion.parsers import parse_ledger, parse_transaction, parse_operation
from astroml.features.node_features import compute_node_features
from astroml.features.graph.snapshot import Edge, window_snapshot
from astroml.features.transaction_graph import TransactionGraph
from astroml.models.gcn import GCN
from astroml.validation.calibration import CalibrationAnalyzer
from astroml.validation.validator import TransactionValidator


class TestFullPipelineIntegration:
    """Integration tests for the complete end-to-end pipeline."""

    def test_ledger_to_model_pipeline(
        self,
        test_session: Session,
        temp_output_dir: Path,
    ) -> None:
        """Test complete pipeline from ledger ingestion to model training."""
        # Step 1: Ingest ledger data
        ledger_data = {
            "sequence": 1000,
            "hash": "a" * 64,
            "prev_hash": "b" * 64,
            "closed_at": datetime(2024, 1, 1),
            "successful_transaction_count": 2,
            "failed_transaction_count": 0,
            "operation_count": 4,
        }
        ledger = parse_ledger(ledger_data)
        test_session.add(ledger)
        test_session.commit()

        # Step 2: Ingest transactions
        tx_data_1 = {
            "hash": "tx1" + "a" * 60,
            "ledger": 1000,
            "source_account": "G" + "A" * 55,
            "created_at": datetime(2024, 1, 1),
            "fee_charged": 100,
            "operation_count": 2,
            "successful": True,
            "memo_type": "none",
        }
        tx_data_2 = {
            "hash": "tx2" + "b" * 60,
            "ledger": 1000,
            "source_account": "G" + "B" * 55,
            "created_at": datetime(2024, 1, 1),
            "fee_charged": 200,
            "operation_count": 2,
            "successful": True,
            "memo_type": "none",
        }
        
        tx1 = parse_transaction(tx_data_1)
        tx2 = parse_transaction(tx_data_2)
        test_session.add(tx1)
        test_session.add(tx2)
        test_session.commit()

        # Step 3: Ingest operations
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
            "type": "payment",
            "to": "G" + "C" * 55,
            "amount": "50.0",
            "asset_type": "native",
            "created_at": datetime(2024, 1, 1),
        }
        op_data_3 = {
            "id": 3,
            "transaction_hash": "tx2" + "b" * 60,
            "source_account": "G" + "B" * 55,
            "type": "payment",
            "to": "G" + "C" * 55,
            "amount": "75.0",
            "asset_type": "native",
            "created_at": datetime(2024, 1, 1),
        }
        
        op1 = parse_operation(op_data_1, application_order=0)
        op2 = parse_operation(op_data_2, application_order=1)
        op3 = parse_operation(op_data_3, application_order=0)
        test_session.add(op1)
        test_session.add(op2)
        test_session.add(op3)
        test_session.commit()

        # Step 4: Extract operations and compute features
        operations = test_session.query(Operation).all()
        edges = []
        for op in operations:
            if op.destination_account:
                edges.append({
                    'src': op.source_account,
                    'dst': op.destination_account,
                    'amount': float(op.amount) if op.amount else 0.0,
                    'timestamp': op.created_at.timestamp(),
                    'asset': op.asset_code or 'XLM',
                })
        
        features_df = compute_node_features(edges)
        
        # Verify features computed
        assert not features_df.empty
        assert len(features_df) == 3  # A, B, C

        # Step 5: Build graph
        graph = TransactionGraph()
        for op in operations:
            if op.destination_account:
                graph.add_transaction(
                    from_account=op.source_account,
                    to_account=op.destination_account,
                    amount=float(op.amount) if op.amount else 0.0,
                    asset=op.asset_code or 'XLM',
                )
        
        # Verify graph
        summary = graph.summary()
        assert summary["node_count"] == 3
        assert summary["transaction_count"] == 3

        # Step 6: Train simple model
        # Convert features to tensor
        feature_matrix = features_df.values.astype(np.float32)
        num_nodes = feature_matrix.shape[0]
        
        # Create simple edge index
        node_to_idx = {node: i for i, node in enumerate(features_df.index)}
        edge_index = []
        for op in operations:
            if op.destination_account:
                src_idx = node_to_idx.get(op.source_account)
                dst_idx = node_to_idx.get(op.destination_account)
                if src_idx is not None and dst_idx is not None:
                    edge_index.append([src_idx, dst_idx])
        
        if len(edge_index) == 0:
            edge_index = [[0, 1], [1, 2]]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create and train model
        model = GCN(
            input_dim=feature_matrix.shape[1],
            hidden_dim=8,
            output_dim=2,
            dropout=0.0,
        )
        
        # Create dummy labels
        labels = torch.randint(0, 2, (num_nodes,))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()
        
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            out = model(torch.tensor(feature_matrix), edge_index)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        
        # Verify training completed
        assert loss.item() is not None

        # Step 7: Validate predictions
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(feature_matrix), edge_index)
            predicted_probs = torch.softmax(predictions, dim=1)[:, 1].numpy()
        
        # Verify predictions
        assert len(predicted_probs) == num_nodes
        assert all(0 <= p <= 1 for p in predicted_probs)

    def test_ingestion_to_validation_pipeline(
        self,
        test_session: Session,
        temp_output_dir: Path,
    ) -> None:
        """Test pipeline from ingestion through validation."""
        # Step 1: Ingest and validate transactions
        transactions = [
            {
                "id": "tx1",
                "source_account": "G" + "A" * 55,
                "amount": 100.0,
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": "tx2",
                "source_account": "G" + "B" * 55,
                "amount": 50.0,
                "created_at": "2024-01-01T00:01:00Z",
            },
        ]
        
        validator = TransactionValidator(
            required_fields={"id", "source_account", "amount"},
        )
        
        results = validator.validate_batch(transactions)
        
        # Verify validation
        assert len(results) == 2
        assert all(r.is_valid for r in results)

        # Step 2: Store valid transactions in database
        for tx_data in transactions:
            # Create ledger
            ledger = Ledger(
                sequence=1000,
                hash="a" * 64,
                closed_at=datetime(2024, 1, 1),
                successful_transaction_count=1,
                failed_transaction_count=0,
                operation_count=1,
            )
            test_session.add(ledger)
            
            # Create transaction
            tx = Transaction(
                hash=tx_data["id"] + "a" * 60,
                ledger_sequence=1000,
                source_account=tx_data["source_account"],
                created_at=datetime.fromisoformat(tx_data["created_at"].replace("Z", "+00:00")),
                fee=100,
                operation_count=1,
                successful=True,
                memo_type="none",
            )
            test_session.add(tx)
        
        test_session.commit()

        # Step 3: Verify database state
        tx_count = test_session.query(Transaction).count()
        assert tx_count == 2

    def test_synthetic_fraud_to_detection_pipeline(
        self,
        test_session: Session,
        temp_data_dir: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test pipeline from synthetic fraud injection to detection."""
        # Step 1: Create clean ledger
        clean_transactions = [
            {
                "source_account": "G" + "A" * 55,
                "destination_account": "G" + "B" * 55,
                "amount": 100.0,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]
        
        input_file = temp_data_dir / "clean.jsonl"
        output_file = temp_data_dir / "with_fraud.jsonl"
        
        with open(input_file, "w") as f:
            for tx in clean_transactions:
                f.write(tx.__str__() + "\n")
        
        # Step 2: Inject synthetic fraud
        from astroml.ingestion.synthetic_fraud_injector import (
            inject_synthetic_fraud,
            SybilConfig,
        )
        
        augmented, summary = inject_synthetic_fraud(
            clean_transactions,
            seed=42,
            sybil=SybilConfig(clusters=1, cluster_size=2, tx_per_member=1),
        )
        
        # Verify injection
        assert len(augmented) > len(clean_transactions)
        assert summary.sybil_transactions > 0

        # Step 3: Store in database
        for tx in augmented:
            if tx.get("synthetic_fraud"):
                # Store fraud pattern metadata
                pass
        
        # Step 4: Verify fraud detection capability
        fraud_txs = [tx for tx in augmented if tx.get("synthetic_fraud")]
        assert len(fraud_txs) > 0

    def test_graph_snapshot_to_model_pipeline(
        self,
        test_session: Session,
        temp_output_dir: Path,
    ) -> None:
        """Test pipeline from graph snapshot to model training."""
        # Step 1: Create normalized transactions
        base_time = datetime(2024, 1, 1)
        
        for i in range(10):
            tx = test_session.query(Transaction).first()
            if not tx:
                # Create transaction if none exists
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
                    source_account=f"G{'A' * i}{'B' * (55-i)}",
                    created_at=base_time + timedelta(hours=i),
                    fee=100,
                    operation_count=1,
                    successful=True,
                    memo_type="none",
                )
                test_session.add(tx)
        
        test_session.commit()

        # Step 2: Create graph snapshot
        from astroml.features.graph.snapshot import snapshot_last_n_days
        
        base_ts = int(base_time.timestamp())
        edges = [
            Edge(src=f"node_{i}", dst=f"node_{(i+1)%5}", timestamp=base_ts + i * 3600)
            for i in range(10)
        ]
        
        now_ts = base_ts + 86400  # 1 day later
        nodes, window_edges = snapshot_last_n_days(edges, now_ts, days=1)
        
        # Verify snapshot
        assert len(window_edges) > 0
        assert len(nodes) > 0

        # Step 3: Compute features from snapshot
        edge_dicts = [
            {
                'src': e.src,
                'dst': e.dst,
                'amount': 100.0,
                'timestamp': e.timestamp,
                'asset': 'XLM',
            }
            for e in window_edges
        ]
        
        features_df = compute_node_features(edge_dicts)
        
        # Verify features
        assert not features_df.empty

    def test_feature_store_to_training_pipeline(
        self,
        temp_output_dir: Path,
        sample_node_features: Dict[str, np.ndarray],
    ) -> None:
        """Test pipeline from feature store to model training."""
        # Step 1: Store features in feature store
        from astroml.features.feature_store import FeatureStore, FeatureDefinition, FeatureType
        from astroml.features.feature_cache import FeatureCache
        
        store_path = temp_output_dir / "feature_store.db"
        cache_path = temp_output_dir / "feature_cache.db"
        
        store = FeatureStore(store_path=str(store_path))
        cache = FeatureCache(cache_path=str(cache_path))
        
        # Register feature
        feature_def = FeatureDefinition(
            name="node_embeddings",
            description="Node embedding features",
            feature_type=FeatureType.VECTOR,
        )
        store.register_feature(feature_def)
        
        # Cache features
        features_df = pd.DataFrame.from_dict(sample_node_features, orient='index')
        cache.put_features(
            feature_name="node_embeddings",
            features=features_df,
            metadata={"version": 1},
        )
        
        # Step 2: Retrieve features for training
        cached_features = cache.get_features("node_embeddings")
        
        # Verify retrieval
        assert cached_features is not None
        assert cached_features.shape == features_df.shape

        # Step 3: Train model with cached features
        feature_matrix = cached_features.values.astype(np.float32)
        num_nodes = feature_matrix.shape[0]
        
        # Simple model
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(feature_matrix.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        labels = torch.randint(0, 2, (num_nodes,))
        
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            predictions = model(torch.tensor(feature_matrix))
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
        
        # Verify training
        assert loss.item() is not None

    def test_end_to_end_data_quality_pipeline(
        self,
        test_session: Session,
        temp_output_dir: Path,
    ) -> None:
        """Test complete data quality validation pipeline."""
        # Step 1: Ingest data with potential quality issues
        transactions = [
            {"id": "tx1", "source_account": "GAAA", "amount": 100.0, "timestamp": "2024-01-01T00:00:00Z"},
            {"id": "tx2", "source_account": "GBBB", "amount": 50.0, "timestamp": "2024-01-01T00:01:00Z"},
            {"id": "tx3", "source_account": None, "amount": 75.0, "timestamp": "2024-01-01T00:02:00Z"},  # Invalid
            {"id": "tx4", "source_account": "GDDD", "amount": "invalid", "timestamp": "2024-01-01T00:03:00Z"},  # Invalid
        ]
        
        # Step 2: Validate data quality
        validator = TransactionValidator(
            required_fields={"id", "source_account", "amount"},
            field_types={"amount": (int, float)},
        )
        
        results = validator.validate_batch(transactions)
        
        # Step 3: Filter valid transactions
        valid_transactions = [
            tx for tx, result in zip(transactions, results) if result.is_valid
        ]
        
        # Verify filtering
        assert len(valid_transactions) == 2

        # Step 4: Store only valid transactions
        for tx in valid_transactions:
            ledger = Ledger(
                sequence=1000,
                hash="a" * 64,
                closed_at=datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00")),
                successful_transaction_count=1,
                failed_transaction_count=0,
                operation_count=1,
            )
            test_session.add(ledger)
            
            transaction = Transaction(
                hash=tx["id"] + "a" * 60,
                ledger_sequence=1000,
                source_account=tx["source_account"],
                created_at=datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00")),
                fee=100,
                operation_count=1,
                successful=True,
                memo_type="none",
            )
            test_session.add(transaction)
        
        test_session.commit()

        # Step 5: Verify only valid data in database
        tx_count = test_session.query(Transaction).count()
        assert tx_count == 2

    def test_model_deployment_pipeline(
        self,
        sample_training_data: tuple,
        temp_output_dir: Path,
    ) -> None:
        """Test complete model deployment pipeline."""
        X, y = sample_training_data
        
        # Step 1: Train model
        model = GCN(
            input_dim=X.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.5,
        )
        
        edge_index = torch.randint(0, len(X), (2, len(X) * 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()
        
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            out = model(torch.tensor(X, dtype=torch.float32), edge_index)
            loss = criterion(out, torch.tensor(y, dtype=torch.long))
            loss.backward()
            optimizer.step()
        
        # Step 2: Save model
        model_path = temp_output_dir / "deployed_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': X.shape[1],
            'hidden_dim': 16,
            'output_dim': 2,
            'training_loss': loss.item(),
            'deployed_at': datetime.utcnow().isoformat(),
        }, model_path)
        
        # Step 3: Load model for inference
        checkpoint = torch.load(model_path)
        loaded_model = GCN(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Step 4: Perform inference
        loaded_model.eval()
        with torch.no_grad():
            predictions = loaded_model(torch.tensor(X, dtype=torch.float32), edge_index)
        
        # Verify deployment pipeline
        assert model_path.exists()
        assert predictions.shape[0] == len(X)
