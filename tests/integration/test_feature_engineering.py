"""Integration tests for the feature engineering pipeline.

These tests verify the complete workflow from database operations
to computed features, including feature store integration and caching.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from astroml.db.schema import Operation, Transaction, Ledger
from astroml.features.node_features import compute_node_features
from astroml.features.feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureType,
    FeatureStatus,
)
from astroml.features.feature_engine import FeatureEngineering as FeatureEngine, ComputationTask, ComputationStatus
from astroml.features.feature_cache import FeatureCache


class TestNodeFeaturesIntegration:
    """Integration tests for node feature computation from database."""

    def test_compute_features_from_database_operations(
        self,
        populated_test_db: Session,
    ) -> None:
        """Test computing node features directly from database operations."""
        # Query operations from database
        operations = populated_test_db.query(Operation).all()
        
        # Convert to edge format
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
        
        # Compute features
        features_df = compute_node_features(edges)
        
        # Verify features were computed
        assert not features_df.empty
        assert 'in_degree' in features_df.columns
        assert 'out_degree' in features_df.columns
        assert 'total_received' in features_df.columns
        assert 'total_sent' in features_df.columns
        assert 'account_age' in features_df.columns
        
        # Verify data types
        assert features_df['in_degree'].dtype == np.int64
        assert features_df['out_degree'].dtype == np.int64
        assert features_df['total_received'].dtype == float
        assert features_df['total_sent'].dtype == float

    def test_compute_features_with_first_seen_provided(
        self,
        populated_test_db: Session,
    ) -> None:
        """Test computing features with externally provided first_seen timestamps."""
        operations = populated_test_db.query(Operation).all()
        
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
        
        # Provide external first_seen data
        base_time = datetime(2024, 1, 1)
        nodes_first_seen = {
            'G' + 'A' * 55: (base_time - timedelta(days=30)).timestamp(),
            'G' + 'B' * 55: (base_time - timedelta(days=15)).timestamp(),
        }
        
        features_df = compute_node_features(
            edges,
            nodes_first_seen=nodes_first_seen,
            ref_time=base_time.timestamp(),
        )
        
        # Verify account age uses provided first_seen where available
        assert 'account_age' in features_df.columns
        assert features_df['account_age'].min() >= 0

    def test_compute_features_with_empty_edges(
        self,
    ) -> None:
        """Test computing features with empty edge list."""
        features_df = compute_node_features([])
        
        # Should return empty DataFrame with correct columns
        assert features_df.empty
        expected_columns = [
            'in_degree', 'out_degree', 'total_received', 'total_sent',
            'account_age', 'first_seen', 'unique_asset_count', 'asset_entropy'
        ]
        assert list(features_df.columns) == expected_columns


class TestFeatureStoreIntegration:
    """Integration tests for feature store with database."""

    def test_register_and_retrieve_feature(
        self,
        test_session: Session,
        temp_data_dir: Path,
    ) -> None:
        """Test registering a feature definition and retrieving it."""
        store_path = temp_data_dir / "feature_store.db"
        store = FeatureStore(store_path=str(store_path))
        
        # Define a simple feature
        def simple_feature(data: pd.DataFrame) -> pd.DataFrame:
            return data[['in_degree', 'out_degree']]
        
        feature_def = FeatureDefinition(
            name="degree_features",
            description="Simple degree features",
            feature_type=FeatureType.NUMERIC,
            computation_function=simple_feature,
            tags=["graph", "basic"],
            owner="ml-team",
            status=FeatureStatus.PRODUCTION,
        )
        
        # Register feature
        store.register_feature(feature_def)
        
        # Retrieve feature
        retrieved = store.get_feature("degree_features", version=1)
        
        assert retrieved is not None
        assert retrieved.name == "degree_features"
        assert retrieved.status == FeatureStatus.PRODUCTION
        assert "graph" in retrieved.tags

    def test_compute_and_cache_features(
        self,
        test_session: Session,
        temp_data_dir: Path,
        sample_node_features: Dict[str, np.ndarray],
    ) -> None:
        """Test computing features and caching them."""
        cache_path = temp_data_dir / "feature_cache.db"
        cache = FeatureCache(cache_path=str(cache_path))
        
        # Create sample feature data
        feature_data = pd.DataFrame.from_dict(sample_node_features, orient='index')
        feature_data.index.name = 'node_id'
        
        # Cache features
        cache.put_features(
            feature_name="test_features",
            features=feature_data,
            metadata={"version": 1, "computed_at": datetime.utcnow().isoformat()},
        )
        
        # Retrieve cached features
        cached = cache.get_features("test_features")
        
        assert cached is not None
        assert cached.shape == feature_data.shape
        assert np.allclose(cached.values, feature_data.values)

    def test_feature_versioning(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test feature versioning in the store."""
        store_path = temp_data_dir / "feature_store.db"
        store = FeatureStore(store_path=str(store_path))
        
        # Register version 1
        feature_v1 = FeatureDefinition(
            name="evolving_feature",
            description="First version",
            feature_type=FeatureType.NUMERIC,
            version=1,
        )
        store.register_feature(feature_v1)
        
        # Register version 2
        feature_v2 = FeatureDefinition(
            name="evolving_feature",
            description="Second version with improvements",
            feature_type=FeatureType.NUMERIC,
            version=2,
        )
        store.register_feature(feature_v2)
        
        # Retrieve both versions
        v1 = store.get_feature("evolving_feature", version=1)
        v2 = store.get_feature("evolving_feature", version=2)
        
        assert v1 is not None
        assert v2 is not None
        assert v1.version == 1
        assert v2.version == 2
        assert v1.description != v2.description

    def test_feature_lineage_tracking(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test tracking feature lineage and dependencies."""
        store_path = temp_data_dir / "feature_store.db"
        store = FeatureStore(store_path=str(store_path))
        
        # Register base feature
        base_feature = FeatureDefinition(
            name="base_transaction_count",
            description="Count of transactions",
            feature_type=FeatureType.NUMERIC,
        )
        store.register_feature(base_feature)
        
        # Register derived feature
        derived_feature = FeatureDefinition(
            name="normalized_transaction_count",
            description="Normalized transaction count",
            feature_type=FeatureType.NUMERIC,
            parameters={"base_feature": "base_transaction_count"},
            metadata={"depends_on": ["base_transaction_count"]},
        )
        store.register_feature(derived_feature)
        
        # Retrieve lineage
        lineage = store.get_feature_lineage("normalized_transaction_count")
        
        assert lineage is not None
        assert "base_transaction_count" in lineage


class TestFeatureEngineIntegration:
    """Integration tests for feature computation engine."""

    def test_execute_computation_task(
        self,
        sample_node_features: Dict[str, np.ndarray],
    ) -> None:
        """Test executing a single feature computation task."""
        engine = FeatureEngine()
        
        # Create sample input data
        input_data = pd.DataFrame.from_dict(sample_node_features, orient='index')
        input_data.index.name = 'node_id'
        
        # Define a simple computation function
        def compute_sum(data: pd.DataFrame) -> pd.DataFrame:
            return data.sum(axis=1).to_frame('feature_sum')
        
        # Create task
        task = ComputationTask(
            task_id="test_task_1",
            feature_name="sum_feature",
            data=input_data,
            parameters={},
        )
        
        # Execute task
        result = engine.execute_task(task, compute_sum)
        
        assert result is not None
        assert result.status == ComputationStatus.COMPLETED
        assert result.result is not None
        assert 'feature_sum' in result.result.columns

    def test_parallel_feature_computation(
        self,
        sample_node_features: Dict[str, np.ndarray],
    ) -> None:
        """Test parallel computation of multiple features."""
        engine = FeatureEngine(max_workers=2)
        
        input_data = pd.DataFrame.from_dict(sample_node_features, orient='index')
        input_data.index.name = 'node_id'
        
        # Define multiple computation functions
        def compute_mean(data: pd.DataFrame) -> pd.DataFrame:
            return data.mean(axis=1).to_frame('feature_mean')
        
        def compute_std(data: pd.DataFrame) -> pd.DataFrame:
            return data.std(axis=1).to_frame('feature_std')
        
        # Create tasks
        tasks = [
            ComputationTask(
                task_id=f"task_{i}",
                feature_name=f"feature_{i}",
                data=input_data,
            )
            for i in range(2)
        ]
        
        # Execute in parallel
        results = engine.execute_parallel(
            tasks,
            [compute_mean, compute_std],
        )
        
        assert len(results) == 2
        assert all(r.status == ComputationStatus.COMPLETED for r in results)
        assert all(r.result is not None for r in results)

    def test_feature_dependency_resolution(
        self,
        sample_node_features: Dict[str, np.ndarray],
    ) -> None:
        """Test resolving feature dependencies during computation."""
        engine = FeatureEngine()
        
        input_data = pd.DataFrame.from_dict(sample_node_features, orient='index')
        
        # Define dependent features
        def base_feature(data: pd.DataFrame) -> pd.DataFrame:
            return data.iloc[:, :2].copy()
        
        def derived_feature(data: pd.DataFrame) -> pd.DataFrame:
            # Depends on base_feature output
            return data.sum(axis=1).to_frame('derived')
        
        # Create tasks with dependencies
        base_task = ComputationTask(
            task_id="base_task",
            feature_name="base_feature",
            data=input_data,
        )
        
        derived_task = ComputationTask(
            task_id="derived_task",
            feature_name="derived_feature",
            data=input_data,  # Will be replaced with base_task result
        )
        
        # Execute base task
        base_result = engine.execute_task(base_task, base_feature)
        
        # Execute derived task with base result as input
        derived_result = engine.execute_task(
            derived_task,
            derived_feature,
            input_data=base_result.result,
        )
        
        assert base_result.status == ComputationStatus.COMPLETED
        assert derived_result.status == ComputationStatus.COMPLETED


class TestEndToEndFeaturePipeline:
    """Integration tests for complete feature engineering pipeline."""

    def test_database_to_features_pipeline(
        self,
        populated_test_db: Session,
        temp_data_dir: Path,
    ) -> None:
        """Test complete pipeline from database to computed features."""
        # Step 1: Extract operations from database
        operations = populated_test_db.query(Operation).all()
        
        # Step 2: Convert to edge format
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
        
        # Step 3: Compute node features
        features_df = compute_node_features(edges)
        
        # Step 4: Cache features
        cache_path = temp_data_dir / "feature_cache.db"
        cache = FeatureCache(cache_path=str(cache_path))
        cache.put_features(
            feature_name="node_features",
            features=features_df,
            metadata={"source": "database", "computed_at": datetime.utcnow().isoformat()},
        )
        
        # Step 5: Retrieve cached features
        cached_features = cache.get_features("node_features")
        
        # Verify pipeline
        assert not features_df.empty
        assert cached_features is not None
        assert cached_features.equals(features_df)

    def test_feature_store_workflow(
        self,
        temp_data_dir: Path,
        sample_node_features: Dict[str, np.ndarray],
    ) -> None:
        """Test complete feature store workflow."""
        store_path = temp_data_dir / "feature_store.db"
        store = FeatureStore(store_path=str(store_path))
        
        # Step 1: Register feature definition
        def aggregate_features(data: pd.DataFrame) -> pd.DataFrame:
            return data.agg(['mean', 'std']).T
        
        feature_def = FeatureDefinition(
            name="aggregate_stats",
            description="Aggregate statistics for node features",
            feature_type=FeatureType.NUMERIC,
            computation_function=aggregate_features,
            status=FeatureStatus.PRODUCTION,
        )
        store.register_feature(feature_def)
        
        # Step 2: Prepare input data
        input_data = pd.DataFrame.from_dict(sample_node_features, orient='index')
        
        # Step 3: Compute feature
        computed = feature_def.computation_function(input_data)
        
        # Step 4: Store computed feature
        cache_path = temp_data_dir / "feature_cache.db"
        cache = FeatureCache(cache_path=str(cache_path))
        cache.put_features(
            feature_name="aggregate_stats",
            features=computed,
            metadata={"feature_id": feature_def.feature_id},
        )
        
        # Step 5: Retrieve and verify
        retrieved = cache.get_features("aggregate_stats")
        
        assert retrieved is not None
        assert not retrieved.empty
        assert 'mean' in retrieved.columns or 'std' in retrieved.columns

    def test_incremental_feature_update(
        self,
        populated_test_db: Session,
        temp_data_dir: Path,
    ) -> None:
        """Test incremental feature updates as new data arrives."""
        cache_path = temp_data_dir / "feature_cache.db"
        cache = FeatureCache(cache_path=str(cache_path))
        
        # Initial computation
        operations = populated_test_db.query(Operation).limit(2).all()
        edges = [
            {
                'src': op.source_account,
                'dst': op.destination_account,
                'amount': float(op.amount) if op.amount else 0.0,
                'timestamp': op.created_at.timestamp(),
                'asset': op.asset_code or 'XLM',
            }
            for op in operations
            if op.destination_account
        ]
        
        initial_features = compute_node_features(edges)
        cache.put_features("node_features", initial_features)
        
        # Add new operation
        new_op = Operation(
            id=999,
            transaction_hash="tx_new",
            application_order=0,
            type="payment",
            source_account="G" + "X" * 55,
            destination_account="G" + "Y" * 55,
            amount=150.0,
            asset_code="XLM",
            created_at=datetime(2024, 1, 2),
        )
        populated_test_db.add(new_op)
        populated_test_db.commit()
        
        # Recompute with new data
        all_operations = populated_test_db.query(Operation).all()
        edges = [
            {
                'src': op.source_account,
                'dst': op.destination_account,
                'amount': float(op.amount) if op.amount else 0.0,
                'timestamp': op.created_at.timestamp(),
                'asset': op.asset_code or 'XLM',
            }
            for op in all_operations
            if op.destination_account
        ]
        
        updated_features = compute_node_features(edges)
        
        # Verify update
        assert len(updated_features) >= len(initial_features)
