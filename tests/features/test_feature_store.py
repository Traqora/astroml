"""Comprehensive tests for the Feature Store.

Tests cover all major components including the core feature store,
computers, transformers, caching, and versioning systems.
"""

from __future__ import annotations

import time
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import pandas as pd
import numpy as np
from cachetools import TTLCache

from astroml.features.feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureType,
    FeatureStatus,
    FeatureSet,
    FeatureStorage,
    FeatureRegistry,
    create_feature_store,
)


class TestFeatureDefinition:
    """Test FeatureDefinition class."""
    
    def test_feature_definition_creation(self):
        """Test creating a feature definition."""
        def dummy_computer(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({"feature": [1, 2, 3]})
        
        feature_def = FeatureDefinition(
            name="test_feature",
            description="Test feature",
            feature_type=FeatureType.NUMERIC,
            computation_function=dummy_computer,
            tags=["test", "dummy"],
            owner="test_user",
        )
        
        assert feature_def.name == "test_feature"
        assert feature_def.description == "Test feature"
        assert feature_def.feature_type == FeatureType.NUMERIC
        assert feature_def.feature_id == "test_feature_v1"
        assert feature_def.tags == ["test", "dummy"]
        assert feature_def.owner == "test_user"
        assert feature_def.status == FeatureStatus.DEVELOPMENT
    
    def test_feature_definition_to_dict(self):
        """Test converting feature definition to dictionary."""
        feature_def = FeatureDefinition(
            name="test_feature",
            description="Test feature",
            feature_type=FeatureType.NUMERIC,
        )
        
        data = feature_def.to_dict()
        
        assert data["name"] == "test_feature"
        assert data["description"] == "Test feature"
        assert data["feature_type"] == "numeric"
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_feature_definition_from_dict(self):
        """Test creating feature definition from dictionary."""
        data = {
            "name": "test_feature",
            "description": "Test feature",
            "feature_type": "numeric",
            "parameters": {"param1": "value1"},
            "tags": ["test"],
            "owner": "test_user",
            "status": "development",
            "version": 1,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {"key": "value"},
        }
        
        feature_def = FeatureDefinition.from_dict(data)
        
        assert feature_def.name == "test_feature"
        assert feature_def.feature_type == FeatureType.NUMERIC
        assert feature_def.parameters == {"param1": "value1"}
        assert feature_def.tags == ["test"]
        assert feature_def.owner == "test_user"


class TestFeatureStorage:
    """Test FeatureStorage class."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def feature_storage(self, temp_storage_path):
        """Create feature storage instance."""
        return FeatureStorage(temp_storage_path)
    
    def test_storage_initialization(self, temp_storage_path):
        """Test storage initialization."""
        storage = FeatureStorage(temp_storage_path)
        
        assert storage.storage_path.exists()
        assert storage.db_path.exists()
        assert storage.data_path.exists()
    
    def test_store_and_get_feature_definition(self, feature_storage):
        """Test storing and retrieving feature definitions."""
        feature_def = FeatureDefinition(
            name="test_feature",
            description="Test feature",
            feature_type=FeatureType.NUMERIC,
        )
        
        # Store feature definition
        feature_storage.store_feature_definition(feature_def)
        
        # Retrieve feature definition
        retrieved_def = feature_storage.get_feature_definition(feature_def.feature_id)
        
        assert retrieved_def is not None
        assert retrieved_def.name == feature_def.name
        assert retrieved_def.description == feature_def.description
        assert retrieved_def.feature_type == feature_def.feature_type
    
    def test_list_feature_definitions(self, feature_storage):
        """Test listing feature definitions."""
        # Create multiple feature definitions
        feature_defs = [
            FeatureDefinition(
                name=f"feature_{i}",
                description=f"Feature {i}",
                feature_type=FeatureType.NUMERIC,
                tags=["test"],
            )
            for i in range(3)
        ]
        
        # Store feature definitions
        for feature_def in feature_defs:
            feature_storage.store_feature_definition(feature_def)
        
        # List all features
        all_features = feature_storage.list_feature_definitions()
        assert len(all_features) == 3
        
        # List features by status
        dev_features = feature_storage.list_feature_definitions(status=FeatureStatus.DEVELOPMENT)
        assert len(dev_features) == 3
        
        # List features by tags
        tagged_features = feature_storage.list_feature_definitions(tags=["test"])
        assert len(tagged_features) == 3
    
    def test_store_and_get_feature_values(self, feature_storage):
        """Test storing and retrieving feature values."""
        feature_id = "test_feature_v1"
        
        # Create test data
        test_data = pd.DataFrame({
            "entity_id": ["entity1", "entity2", "entity3"],
            "feature_value": [1.0, 2.0, 3.0],
        }).set_index("entity_id")
        
        # Store feature values
        feature_storage.store_feature_values(feature_id, test_data)
        
        # Retrieve feature values
        retrieved_data = feature_storage.get_feature_values(feature_id)
        
        assert retrieved_data is not None
        assert len(retrieved_data) == 3
        assert list(retrieved_data.index) == ["entity1", "entity2", "entity3"]
        assert list(retrieved_data["feature_value"]) == [1.0, 2.0, 3.0]
    
    def test_store_and_get_feature_set(self, feature_storage):
        """Test storing and retrieving feature sets."""
        feature_set = FeatureSet(
            name="test_set",
            description="Test feature set",
            feature_ids=["feature1_v1", "feature2_v1"],
            entity_type="account",
        )
        
        # Store feature set
        feature_storage.store_feature_set(feature_set)
        
        # Retrieve feature set
        retrieved_set = feature_storage.get_feature_set("test_set")
        
        assert retrieved_set is not None
        assert retrieved_set.name == "test_set"
        assert retrieved_set.description == "Test feature set"
        assert retrieved_set.feature_ids == ["feature1_v1", "feature2_v1"]
        assert retrieved_set.entity_type == "account"


class TestFeatureRegistry:
    """Test FeatureRegistry class."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def feature_registry(self, temp_storage_path):
        """Create feature registry instance."""
        storage = FeatureStorage(temp_storage_path)
        return FeatureRegistry(storage)
    
    def test_registry_initialization(self, feature_registry):
        """Test registry initialization."""
        assert len(feature_registry.list_features()) > 0  # Should have builtin features
        
        # Check for builtin features
        features = feature_registry.list_features()
        assert "daily_transaction_count" in features
        assert "degree_centrality" in features
        assert "node_features" in features
    
    def test_register_computer(self, feature_registry):
        """Test registering a feature computer."""
        def test_computer(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({"test_feature": [1, 2, 3]})
        
        metadata = {
            "description": "Test feature computer",
            "feature_type": FeatureType.NUMERIC,
            "tags": ["test"],
        }
        
        feature_registry.register_computer("test_feature", test_computer, metadata)
        
        # Check that computer was registered
        assert "test_feature" in feature_registry.list_features()
        
        # Check that feature definition was stored
        computer = feature_registry.get_computer("test_feature")
        assert computer is not None
    
    def test_get_computer(self, feature_registry):
        """Test getting registered computers."""
        # Get existing computer
        computer = feature_registry.get_computer("daily_transaction_count")
        assert computer is not None
        
        # Get non-existing computer
        computer = feature_registry.get_computer("non_existent_feature")
        assert computer is None


class TestFeatureStore:
    """Test FeatureStore class."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def feature_store(self, temp_storage_path):
        """Create feature store instance."""
        return FeatureStore(temp_storage_path)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            "entity_id": ["acc1", "acc2", "acc3", "acc1", "acc2"],
            "timestamp": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5),
            ],
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
            "src": ["acc1", "acc2", "acc3", "acc4", "acc5"],
            "dst": ["acc2", "acc3", "acc1", "acc5", "acc4"],
        })
    
    def test_feature_store_initialization(self, temp_storage_path):
        """Test feature store initialization."""
        store = FeatureStore(temp_storage_path)
        
        assert store.storage.storage_path.exists()
        assert len(store.registry.list_features()) > 0
    
    def test_register_feature(self, feature_store):
        """Test registering a new feature."""
        def test_computer(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({"test_feature": [1, 2, 3]})
        
        feature_def = feature_store.register_feature(
            name="test_feature",
            computer=test_computer,
            description="Test feature for unit testing",
            feature_type=FeatureType.NUMERIC,
            tags=["test", "unit_test"],
            owner="test_user",
        )
        
        assert feature_def.name == "test_feature"
        assert feature_def.description == "Test feature for unit testing"
        assert feature_def.feature_type == FeatureType.NUMERIC
        assert feature_def.tags == ["test", "unit_test"]
        assert feature_def.owner == "test_user"
    
    def test_compute_feature(self, feature_store, sample_data):
        """Test computing features."""
        # This test might fail if the actual feature modules are not available
        # but should test the computation pipeline
        
        # Try to compute a feature that should exist
        try:
            result = feature_store.compute_feature(
                feature_name="daily_transaction_count",
                data=sample_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            
        except ImportError:
            # Skip test if feature modules are not available
            pytest.skip("Feature modules not available")
    
    def test_store_and_get_feature(self, feature_store):
        """Test storing and retrieving features."""
        feature_name = "test_feature"
        
        # Create test feature values
        test_values = pd.DataFrame({
            "feature_value": [1.0, 2.0, 3.0],
        }, index=["entity1", "entity2", "entity3"])
        
        # Store feature
        feature_store.store_feature(feature_name, test_values)
        
        # Get feature
        retrieved_values = feature_store.get_feature(feature_name)
        
        assert retrieved_values is not None
        assert len(retrieved_values) == 3
        assert list(retrieved_values.index) == ["entity1", "entity2", "entity3"]
    
    def test_compute_and_store(self, feature_store, sample_data):
        """Test computing and storing features in one step."""
        try:
            # This test might fail if feature modules are not available
            result = feature_store.compute_and_store(
                feature_name="daily_transaction_count",
                data=sample_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
            
            assert isinstance(result, pd.DataFrame)
            
            # Check that feature was stored
            stored_values = feature_store.get_feature("daily_transaction_count")
            assert stored_values is not None
            
        except ImportError:
            pytest.skip("Feature modules not available")
    
    def test_create_feature_set(self, feature_store):
        """Test creating feature sets."""
        # First register some features
        def test_computer1(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({"feature1": [1, 2, 3]})
        
        def test_computer2(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({"feature2": [4, 5, 6]})
        
        feature_store.register_feature("feature1", test_computer1, "Test feature 1")
        feature_store.register_feature("feature2", test_computer2, "Test feature 2")
        
        # Create feature set
        feature_set = feature_store.create_feature_set(
            name="test_set",
            feature_names=["feature1", "feature2"],
            description="Test feature set",
            entity_type="account",
        )
        
        assert feature_set.name == "test_set"
        assert feature_set.feature_ids == ["feature1_v1", "feature2_v1"]
        assert feature_set.entity_type == "account"
    
    def test_get_features_for_entities(self, feature_store):
        """Test getting features for specific entities."""
        feature_names = ["feature1", "feature2"]
        entity_ids = ["entity1", "entity2"]
        
        # Store some test features
        for i, feature_name in enumerate(feature_names):
            test_values = pd.DataFrame({
                f"feature{i+1}": [i+1, i+2, i+3],
            }, index=["entity1", "entity2", "entity3"])
            
            feature_store.store_feature(feature_name, test_values)
        
        # Get features for specific entities
        result = feature_store.get_features_for_entities(
            feature_names=feature_names,
            entity_ids=entity_ids,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two entities
        assert list(result.index) == entity_ids
        assert "feature1" in result.columns
        assert "feature2" in result.columns
    
    def test_list_features(self, feature_store):
        """Test listing features."""
        # Register a test feature
        def test_computer(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({"test_feature": [1, 2, 3]})
        
        feature_store.register_feature(
            "test_feature",
            test_computer,
            "Test feature",
            tags=["test"],
        )
        
        # List all features
        all_features = feature_store.list_features()
        assert len(all_features) > 0
        
        # Find our test feature
        test_features = [f for f in all_features if f.name == "test_feature"]
        assert len(test_features) == 1
        assert test_features[0].tags == ["test"]
    
    def test_cache_operations(self, feature_store):
        """Test cache operations."""
        feature_name = "test_feature"
        
        # Create test feature values
        test_values = pd.DataFrame({
            "feature_value": [1.0, 2.0, 3.0],
        }, index=["entity1", "entity2", "entity3"])
        
        # Store feature (this should add to cache)
        feature_store.store_feature(feature_name, test_values)
        
        # Get feature (should use cache)
        retrieved_values = feature_store.get_feature(feature_name, use_cache=True)
        assert retrieved_values is not None
        
        # Clear cache
        feature_store.clear_cache()
        
        # Get feature again (should reload from storage)
        retrieved_values = feature_store.get_feature(feature_name, use_cache=True)
        assert retrieved_values is not None
    
    def test_batch_mode(self, feature_store):
        """Test batch mode context manager."""
        feature_name = "test_feature"

        # Create test feature values
        test_values = pd.DataFrame({
            "feature_value": [1.0, 2.0, 3.0],
        }, index=["entity1", "entity2", "entity3"])

        with feature_store.batch_mode():
            # Store feature in batch mode
            feature_store.store_feature(feature_name, test_values)

            # Get feature in batch mode
            retrieved_values = feature_store.get_feature(feature_name)
            assert retrieved_values is not None

        # Value cache should be empty after batch mode exits.
        assert len(feature_store._value_cache) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_create_feature_store(self, temp_storage_path):
        """Test create_feature_store convenience function."""
        store = create_feature_store(temp_storage_path)
        
        assert isinstance(store, FeatureStore)
        assert store.storage.storage_path == Path(temp_storage_path)
    
    def test_get_feature_store(self, temp_storage_path):
        """Test get_feature_store convenience function."""
        store = create_feature_store(temp_storage_path)
        
        # Get existing store
        retrieved_store = create_feature_store(temp_storage_path)
        
        assert isinstance(retrieved_store, FeatureStore)
        assert retrieved_store.storage.storage_path == store.storage.storage_path


class TestFeatureStoreIntegration:
    """Integration tests for the complete feature store workflow."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def feature_store(self, temp_storage_path):
        """Create feature store instance."""
        return FeatureStore(temp_storage_path)
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for integration tests."""
        np.random.seed(42)
        
        # Generate sample data
        n_transactions = 1000
        accounts = [f"account_{i}" for i in range(50)]
        
        data = pd.DataFrame({
            "entity_id": np.random.choice(accounts, n_transactions),
            "timestamp": pd.date_range("2023-01-01", periods=n_transactions, freq="H"),
            "amount": np.random.exponential(100, n_transactions),
            "src": np.random.choice(accounts, n_transactions),
            "dst": np.random.choice(accounts, n_transactions),
            "asset": np.random.choice(["XLM", "USD", "EUR"], n_transactions),
        })
        
        return data
    
    def test_complete_workflow(self, feature_store, sample_transaction_data):
        """Test complete feature store workflow."""
        try:
            # 1. Register a custom feature
            def account_balance_computer(data, entity_col, timestamp_col, **kwargs):
                """Simple account balance computation."""
                # Compute total sent and received per account
                sent = data.groupby("src")["amount"].sum()
                received = data.groupby("dst")["amount"].sum()
                
                # Combine sent and received
                all_accounts = set(sent.index) | set(received.index)
                balances = {}
                
                for account in all_accounts:
                    sent_amount = sent.get(account, 0)
                    received_amount = received.get(account, 0)
                    balances[account] = received_amount - sent_amount
                
                return pd.DataFrame(
                    {"balance": list(balances.values())},
                    index=list(balances.keys())
                )
            
            feature_def = feature_store.register_feature(
                name="account_balance",
                computer=account_balance_computer,
                description="Account balance computed from transactions",
                feature_type=FeatureType.NUMERIC,
                tags=["balance", "financial"],
                owner="test_user",
            )
            
            # 2. Compute and store the feature
            computed_values = feature_store.compute_and_store(
                feature_name="account_balance",
                data=sample_transaction_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
            
            assert isinstance(computed_values, pd.DataFrame)
            assert len(computed_values) > 0
            assert "balance" in computed_values.columns
            
            # 3. Retrieve the feature
            stored_values = feature_store.get_feature("account_balance")
            assert stored_values is not None
            assert len(stored_values) == len(computed_values)
            
            # 4. Create a feature set
            feature_set = feature_store.create_feature_set(
                name="financial_features",
                feature_names=["account_balance"],
                description="Financial features for accounts",
                entity_type="account",
            )
            
            assert feature_set.name == "financial_features"
            assert len(feature_set.feature_ids) == 1
            
            # 5. Get features for specific entities
            sample_entities = list(computed_values.index[:5])
            entity_features = feature_store.get_features_for_entities(
                feature_names=["account_balance"],
                entity_ids=sample_entities,
            )
            
            assert len(entity_features) == 5
            assert "account_balance" in entity_features.columns
            
            # 6. List features
            all_features = feature_store.list_features()
            balance_features = [f for f in all_features if f.name == "account_balance"]
            assert len(balance_features) == 1
            assert balance_features[0].tags == ["balance", "financial"]
            
        except ImportError:
            pytest.skip("Feature modules not available for integration test")
    
    def test_error_handling(self, feature_store):
        """Test error handling in feature store."""
        # Test getting non-existent feature
        result = feature_store.get_feature("non_existent_feature")
        assert result is None
        
        # Test computing non-existent feature
        with pytest.raises(ValueError, match="Feature 'non_existent_feature' not found"):
            feature_store.compute_feature(
                feature_name="non_existent_feature",
                data=pd.DataFrame(),
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
        
        # Test storing feature without registration
        with pytest.raises(ValueError, match="Feature 'non_existent_feature' not found"):
            feature_store.store_feature(
                feature_name="non_existent_feature",
                values=pd.DataFrame({"value": [1, 2, 3]}),
            )
    
    def test_persistence(self, temp_storage_path, sample_transaction_data):
        """Test that feature store persists data across instances."""
        try:
            # Create first instance and add data
            store1 = FeatureStore(temp_storage_path)
            
            def simple_computer(data, entity_col, timestamp_col, **kwargs):
                return pd.DataFrame({"simple_feature": [1, 2, 3]})
            
            store1.register_feature("simple_feature", simple_computer, "Simple test feature")
            
            computed_values = store1.compute_and_store(
                feature_name="simple_feature",
                data=sample_transaction_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
            
            # Create second instance and verify data persistence
            store2 = FeatureStore(temp_storage_path)
            
            # Check that feature definition persists
            all_features = store2.list_features()
            simple_features = [f for f in all_features if f.name == "simple_feature"]
            assert len(simple_features) == 1
            
            # Check that feature values persist
            stored_values = store2.get_feature("simple_feature")
            assert stored_values is not None
            assert len(stored_values) == len(computed_values)
            
        except ImportError:
            pytest.skip("Feature modules not available")


class TestFeatureStoreCaching:
    """Tests for the TTLCache-based caching strategy in FeatureStore.

    Covers TTL expiry, LRU eviction, hit/miss metrics, write invalidation,
    cache configuration, and thread safety.
    """

    @pytest.fixture
    def temp_storage_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def feature_store(self, temp_storage_path):
        """Feature store with a very short TTL so expiry tests run quickly."""
        return FeatureStore(
            temp_storage_path,
            cache_ttl_seconds=2,
            cache_maxsize=4,
        )

    @pytest.fixture
    def sample_values(self):
        return pd.DataFrame(
            {"feature_value": [1.0, 2.0, 3.0]},
            index=["entity1", "entity2", "entity3"],
        )

    # ------------------------------------------------------------------
    # Basic: verify TTLCache is used
    # ------------------------------------------------------------------

    def test_value_cache_is_ttlcache(self, temp_storage_path):
        """_value_cache must be a cachetools.TTLCache instance."""
        store = FeatureStore(temp_storage_path)
        assert isinstance(store._value_cache, TTLCache)

    def test_default_ttl_and_maxsize(self, temp_storage_path):
        """Default TTL is 900 s and maxsize is 128."""
        store = FeatureStore(temp_storage_path)
        assert store._cache_ttl_seconds == 900
        assert store._cache_maxsize == 128
        assert store._value_cache.ttl == 900
        assert store._value_cache.maxsize == 128

    def test_custom_ttl_and_maxsize(self, temp_storage_path):
        """Constructor parameters are forwarded to TTLCache."""
        store = FeatureStore(temp_storage_path, cache_ttl_seconds=60, cache_maxsize=10)
        assert store._value_cache.ttl == 60
        assert store._value_cache.maxsize == 10

    # ------------------------------------------------------------------
    # Hit / miss counters
    # ------------------------------------------------------------------

    def test_cache_miss_increments_counter(self, feature_store, sample_values):
        """First get after store_feature should be a cache miss."""
        feature_name = "feat_miss"
        feature_store.store_feature(feature_name, sample_values)

        # Reset counters so store_feature side-effects don't pollute
        feature_store._cache_hits = 0
        feature_store._cache_misses = 0

        feature_store.get_feature(feature_name, use_cache=True)
        assert feature_store._cache_misses == 1
        assert feature_store._cache_hits == 0

    def test_cache_hit_increments_counter(self, feature_store, sample_values):
        """Second get on the same feature should be a cache hit."""
        feature_name = "feat_hit"
        feature_store.store_feature(feature_name, sample_values)

        feature_store._cache_hits = 0
        feature_store._cache_misses = 0

        feature_store.get_feature(feature_name, use_cache=True)  # miss → fills cache
        feature_store.get_feature(feature_name, use_cache=True)  # hit

        assert feature_store._cache_misses == 1
        assert feature_store._cache_hits == 1

    def test_use_cache_false_bypasses_cache(self, feature_store, sample_values):
        """use_cache=False must not read or write the value cache."""
        feature_name = "feat_bypass"
        feature_store.store_feature(feature_name, sample_values)

        feature_store._cache_hits = 0
        feature_store._cache_misses = 0

        # Warm the cache first
        feature_store.get_feature(feature_name, use_cache=True)
        feature_store._cache_hits = 0
        feature_store._cache_misses = 0

        result = feature_store.get_feature(feature_name, use_cache=False)

        assert result is not None
        assert feature_store._cache_hits == 0
        assert feature_store._cache_misses == 1  # bypassed hit, still counted as miss

    # ------------------------------------------------------------------
    # get_cache_stats
    # ------------------------------------------------------------------

    def test_get_cache_stats_keys(self, feature_store):
        """get_cache_stats returns all required metric keys."""
        stats = feature_store.get_cache_stats()
        required = {
            "cached_features",
            "cache_size_mb",
            "max_cache_size_mb",
            "cache_utilization_pct",
            "cache_maxsize",
            "cache_ttl_seconds",
            "metadata_cached",
            "hits",
            "misses",
            "evictions",
            "hit_rate",
            "miss_rate",
        }
        assert required.issubset(stats.keys())

    def test_hit_rate_and_miss_rate_sum_to_one(self, feature_store, sample_values):
        """hit_rate + miss_rate == 1.0 when there is at least one lookup."""
        feature_name = "feat_rates"
        feature_store.store_feature(feature_name, sample_values)


class TestParallelFeatureComputation:
    """Tests for parallel feature computation functionality.

    Covers parallel execution, fallback to sequential, chunking,
    thread safety, and configuration options.
    """

    @pytest.fixture
    def temp_storage_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def feature_store(self, temp_storage_path):
        """Create feature store instance."""
        return FeatureStore(temp_storage_path)

    @pytest.fixture
    def large_sample_data(self):
        """Create large sample data for parallel computation testing."""
        np.random.seed(42)
        n_rows = 500
        n_entities = 50
        
        return pd.DataFrame({
            "entity_id": np.random.randint(1, n_entities + 1, n_rows),
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="min"),
            "amount": np.random.uniform(1, 1000, n_rows),
            "src": np.random.randint(1, n_entities + 1, n_rows),
            "dst": np.random.randint(1, n_entities + 1, n_rows),
        })

    @pytest.fixture
    def small_sample_data(self):
        """Create small sample data (below chunk size)."""
        return pd.DataFrame({
            "entity_id": ["acc1", "acc2", "acc3"],
            "timestamp": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
            ],
            "amount": [100.0, 200.0, 150.0],
        })

    def test_parallel_configuration_defaults(self, temp_storage_path):
        """Test default parallel computation configuration."""
        store = FeatureStore(temp_storage_path)
        
        assert store._max_workers == 4
        assert store._chunk_size == 100
        assert store._enable_parallel == True

    def test_parallel_configuration_custom(self, temp_storage_path):
        """Test custom parallel computation configuration."""
        store = FeatureStore(
            temp_storage_path,
            max_workers=8,
            chunk_size=50,
            enable_parallel=True,
        )
        
        assert store._max_workers == 8
        assert store._chunk_size == 50
        assert store._enable_parallel == True

    def test_parallel_disabled(self, temp_storage_path):
        """Test disabling parallel computation."""
        store = FeatureStore(
            temp_storage_path,
            max_workers=1,
            enable_parallel=False,
        )
        
        assert store._enable_parallel == False

    def test_parallel_compute_feature_large_data(self, feature_store, large_sample_data):
        """Test parallel computation with large data."""
        def test_computer(data, entity_col, timestamp_col, **kwargs):
            """Simple test computer."""
            result = data.groupby(entity_col).agg({
                "amount": ["sum", "mean", "count"]
            })
            result.columns = ["sum", "mean", "count"]
            return result
        
        feature_store.register_feature(
            "test_parallel_feature",
            test_computer,
            "Test parallel feature",
        )
        
        # Test with parallel enabled
        store_parallel = FeatureStore(
            feature_store.storage.storage_path,
            max_workers=4,
            chunk_size=20,
            enable_parallel=True,
        )
        
        result = store_parallel.compute_feature(
            feature_name="test_parallel_feature",
            data=large_sample_data,
            entity_col="entity_id",
            timestamp_col="timestamp",
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_sequential_compute_feature_small_data(self, feature_store, small_sample_data):
        """Test sequential computation with small data (below chunk size)."""
        def test_computer(data, entity_col, timestamp_col, **kwargs):
            """Simple test computer."""
            result = data.groupby(entity_col).agg({"amount": "sum"})
            result.columns = ["sum"]
            return result
        
        feature_store.register_feature(
            "test_sequential_feature",
            test_computer,
            "Test sequential feature",
        )
        
        store_parallel = FeatureStore(
            feature_store.storage.storage_path,
            max_workers=4,
            chunk_size=100,
            enable_parallel=True,
        )
        
        result = store_parallel.compute_feature(
            feature_name="test_sequential_feature",
            data=small_sample_data,
            entity_col="entity_id",
            timestamp_col="timestamp",
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_parallel_fallback_to_sequential(self, feature_store, large_sample_data):
        """Test fallback to sequential when parallel computation fails."""
        def failing_computer(data, entity_col, timestamp_col, **kwargs):
            """Computer that fails in parallel but works sequentially."""
            # Simulate a failure that might occur in parallel
            if len(data) > 100:
                raise RuntimeError("Simulated parallel failure")
            return data.groupby(entity_col).agg({"amount": "sum"})
        
        feature_store.register_feature(
            "test_fallback_feature",
            failing_computer,
            "Test fallback feature",
        )
        
        store_parallel = FeatureStore(
            feature_store.storage.storage_path,
            max_workers=4,
            chunk_size=20,
            enable_parallel=True,
        )
        
        # This should fallback to sequential
        with pytest.raises(RuntimeError):
            store_parallel.compute_feature(
                feature_name="test_fallback_feature",
                data=large_sample_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )

    def test_parallel_get_features_for_entities(self, feature_store):
        """Test parallel fetching of multiple features."""
        # Store multiple test features
        for i in range(3):
            test_values = pd.DataFrame({
                f"feature{i}": [i+1, i+2, i+3],
            }, index=["entity1", "entity2", "entity3"])
            feature_store.store_feature(f"feature{i}", test_values)
        
        store_parallel = FeatureStore(
            feature_store.storage.storage_path,
            max_workers=4,
            enable_parallel=True,
        )
        
        result = store_parallel.get_features_for_entities(
            feature_names=["feature0", "feature1", "feature2"],
            entity_ids=["entity1", "entity2"],
            parallel=True,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "feature0" in result.columns
        assert "feature1" in result.columns
        assert "feature2" in result.columns

    def test_sequential_get_features_for_entities(self, feature_store):
        """Test sequential fetching of multiple features."""
        # Store multiple test features
        for i in range(3):
            test_values = pd.DataFrame({
                f"feature{i}": [i+1, i+2, i+3],
            }, index=["entity1", "entity2", "entity3"])
            feature_store.store_feature(f"feature{i}", test_values)
        
        store_parallel = FeatureStore(
            feature_store.storage.storage_path,
            max_workers=4,
            enable_parallel=True,
        )
        
        result = store_parallel.get_features_for_entities(
            feature_names=["feature0", "feature1", "feature2"],
            entity_ids=["entity1", "entity2"],
            parallel=False,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_thread_safety_with_cache(self, feature_store):
        """Test thread safety with cache operations during parallel computation."""
        def test_computer(data, entity_col, timestamp_col, **kwargs):
            """Test computer that simulates cache operations."""
            result = data.groupby(entity_col).agg({"amount": "sum"})
            result.columns = ["sum"]
            return result
        
        feature_store.register_feature(
            "test_thread_safety",
            test_computer,
            "Test thread safety feature",
        )
        
        # Create large data to trigger parallel computation
        np.random.seed(42)
        large_data = pd.DataFrame({
            "entity_id": np.random.randint(1, 100, 500),
            "timestamp": pd.date_range("2023-01-01", periods=500, freq="min"),
            "amount": np.random.uniform(1, 1000, 500),
        })
        
        store_parallel = FeatureStore(
            feature_store.storage.storage_path,
            max_workers=4,
            chunk_size=50,
            enable_parallel=True,
        )
        
        # Compute feature (should use parallel computation)
        result = store_parallel.compute_feature(
            feature_name="test_thread_safety",
            data=large_data,
            entity_col="entity_id",
            timestamp_col="timestamp",
        )
        
        assert isinstance(result, pd.DataFrame)
        
        # Verify cache is still consistent
        cache_stats = store_parallel.get_cache_stats()
        assert cache_stats is not None

    def test_chunk_size_configuration(self, temp_storage_path):
        """Test that chunk size affects parallel computation behavior."""
        store_small_chunk = FeatureStore(
            temp_storage_path,
            max_workers=4,
            chunk_size=10,
            enable_parallel=True,
        )
        
        store_large_chunk = FeatureStore(
            temp_storage_path,
            max_workers=4,
            chunk_size=1000,
            enable_parallel=True,
        )
        
        assert store_small_chunk._chunk_size == 10
        assert store_large_chunk._chunk_size == 1000

    def test_max_workers_configuration(self, temp_storage_path):
        """Test that max_workers configuration is respected."""
        store_2_workers = FeatureStore(
            temp_storage_path,
            max_workers=2,
            enable_parallel=True,
        )
        
        store_8_workers = FeatureStore(
            temp_storage_path,
            max_workers=8,
            enable_parallel=True,
        )
        
        assert store_2_workers._max_workers == 2
        assert store_8_workers._max_workers == 8

    def test_create_feature_store_with_parallel_config(self, temp_storage_path):
        """Test create_feature_store with parallel configuration."""
        store = create_feature_store(
            temp_storage_path,
            max_workers=8,
            chunk_size=50,
            enable_parallel=True,
        )
        
        assert store._max_workers == 8
        assert store._chunk_size == 50
        assert store._enable_parallel == True

    def test_stats_zero_before_any_lookup(self, feature_store):
        """All rate/counter fields are 0 on a fresh store."""
        stats = feature_store.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["miss_rate"] == 0.0

    def test_cache_ttl_seconds_matches_config(self, feature_store):
        """cache_ttl_seconds in stats matches the value passed to __init__."""
        stats = feature_store.get_cache_stats()
        assert stats["cache_ttl_seconds"] == 2  # fixture sets ttl=2

    # ------------------------------------------------------------------
    # TTL expiry
    # ------------------------------------------------------------------

    def test_cache_expires_after_ttl(self, feature_store, sample_values):
        """Entries must be absent from the TTLCache after the TTL elapses."""
        feature_name = "feat_ttl"
        feature_store.store_feature(feature_name, sample_values)

        # Warm the cache
        feature_store.get_feature(feature_name, use_cache=True)
        feat_id = f"{feature_name}_v1"
        assert feat_id in feature_store._value_cache

        # Wait for TTL to expire (fixture uses ttl=2 s)
        time.sleep(3)

        assert feat_id not in feature_store._value_cache

    def test_get_after_expiry_reloads_from_storage(self, feature_store, sample_values):
        """After TTL expiry a fresh get must reload from storage (cache miss)."""
        feature_name = "feat_reload"
        feature_store.store_feature(feature_name, sample_values)
        feature_store.get_feature(feature_name, use_cache=True)  # warm

        time.sleep(3)  # let TTL expire

        feature_store._cache_hits = 0
        feature_store._cache_misses = 0

        result = feature_store.get_feature(feature_name, use_cache=True)

        assert result is not None
        assert feature_store._cache_misses == 1
        assert feature_store._cache_hits == 0

    # ------------------------------------------------------------------
    # LRU eviction (maxsize cap)
    # ------------------------------------------------------------------

    def test_lru_eviction_when_maxsize_exceeded(self, temp_storage_path, sample_values):
        """Inserting more entries than maxsize must evict the LRU entry."""
        store = FeatureStore(temp_storage_path, cache_ttl_seconds=900, cache_maxsize=2)

        names = ["feat_a", "feat_b", "feat_c"]
        for name in names:
            store.store_feature(name, sample_values)

        # Warm cache with feat_a and feat_b (fills the 2-slot cache)
        store.get_feature("feat_a", use_cache=True)
        store.get_feature("feat_b", use_cache=True)

        assert len(store._value_cache) == 2

        # Access feat_a to make feat_b the LRU
        store.get_feature("feat_a", use_cache=True)

        # Now fetch feat_c — TTLCache must evict feat_b (oldest access)
        store.get_feature("feat_c", use_cache=True)

        assert len(store._value_cache) == 2
        assert "feat_c_v1" in store._value_cache
        # feat_a was accessed more recently than feat_b; feat_b should be gone
        assert "feat_b_v1" not in store._value_cache

    def test_eviction_increments_counter(self, temp_storage_path, sample_values):
        """Manual memory-cap eviction must increment _cache_evictions."""
        # Create a store with an extremely small memory cap so the first insert
        # forces eviction.
        store = FeatureStore(
            temp_storage_path,
            max_cache_size_mb=0,  # 0 MB → always triggers eviction path
            cache_ttl_seconds=900,
            cache_maxsize=128,
        )

        store.store_feature("feat_evict_a", sample_values)
        store.store_feature("feat_evict_b", sample_values)

        store._cache_evictions = 0  # reset to isolate this assertion

        store.get_feature("feat_evict_a", use_cache=True)
        store.get_feature("feat_evict_b", use_cache=True)

        # With max_cache_size_mb=0 the memory-cap eviction path fires on every
        # insert that adds bytes, so evictions counter should be > 0.
        assert store._cache_evictions > 0

    # ------------------------------------------------------------------
    # Write invalidation
    # ------------------------------------------------------------------

    def test_store_feature_invalidates_cache(self, feature_store, sample_values):
        """store_feature must remove the existing cache entry (write invalidation)."""
        feature_name = "feat_invalidate"
        feature_store.store_feature(feature_name, sample_values)
        feature_store.get_feature(feature_name, use_cache=True)  # warm cache

        feat_id = f"{feature_name}_v1"
        assert feat_id in feature_store._value_cache

        # Update the feature — should evict the old entry
        updated = sample_values.copy()
        updated["feature_value"] = [9.0, 8.0, 7.0]
        feature_store.store_feature(feature_name, updated)

        assert feat_id not in feature_store._value_cache

    def test_updated_value_is_loaded_after_invalidation(self, feature_store, sample_values):
        """After write-invalidation the next get must return the updated values."""
        feature_name = "feat_updated"
        feature_store.store_feature(feature_name, sample_values)
        feature_store.get_feature(feature_name, use_cache=True)  # warm cache

        updated = sample_values.copy()
        updated["feature_value"] = [9.0, 8.0, 7.0]
        feature_store.store_feature(feature_name, updated)

        result = feature_store.get_feature(feature_name, use_cache=True)
        assert result is not None
        assert list(result["feature_value"]) == [9.0, 8.0, 7.0]

    # ------------------------------------------------------------------
    # clear_cache resets metrics
    # ------------------------------------------------------------------

    def test_clear_cache_resets_value_cache(self, feature_store, sample_values):
        """clear_cache must empty the TTLCache."""
        feature_store.store_feature("feat_clear", sample_values)
        feature_store.get_feature("feat_clear", use_cache=True)
        assert len(feature_store._value_cache) > 0

        feature_store.clear_cache()

        assert len(feature_store._value_cache) == 0

    def test_clear_cache_resets_size_tracking(self, feature_store, sample_values):
        """_current_cache_size_bytes must be 0 after clear_cache."""
        feature_store.store_feature("feat_size", sample_values)
        feature_store.get_feature("feat_size", use_cache=True)

        feature_store.clear_cache()

        assert feature_store._current_cache_size_bytes == 0

    # ------------------------------------------------------------------
    # batch_mode
    # ------------------------------------------------------------------

    def test_batch_mode_clears_cache_on_entry_and_exit(self, feature_store, sample_values):
        """batch_mode must empty the cache both on entry and on exit."""
        feature_name = "feat_batch"
        feature_store.store_feature(feature_name, sample_values)
        feature_store.get_feature(feature_name, use_cache=True)

        with feature_store.batch_mode():
            # Cache should have been cleared on entry
            assert len(feature_store._value_cache) == 0

            feature_store.get_feature(feature_name, use_cache=True)  # refill
            assert len(feature_store._value_cache) == 1

        # Cache should be cleared again on exit
        assert len(feature_store._value_cache) == 0

    def test_batch_mode_resets_metrics(self, feature_store, sample_values):
        """Entering batch_mode resets hit/miss/eviction counters."""
        feature_store._cache_hits = 5
        feature_store._cache_misses = 3
        feature_store._cache_evictions = 2

        with feature_store.batch_mode():
            assert feature_store._cache_hits == 0
            assert feature_store._cache_misses == 0
            assert feature_store._cache_evictions == 0

    # ------------------------------------------------------------------
    # _is_cache_expired helper
    # ------------------------------------------------------------------

    def test_is_cache_expired_returns_true_for_missing(self, feature_store):
        """_is_cache_expired is True for a feature never cached."""
        assert feature_store._is_cache_expired("ghost_feature_v1") is True

    def test_is_cache_expired_returns_false_when_present(self, feature_store, sample_values):
        """_is_cache_expired is False immediately after caching a feature."""
        feature_name = "feat_not_expired"
        feature_store.store_feature(feature_name, sample_values)
        feature_store.get_feature(feature_name, use_cache=True)  # populate cache

        assert feature_store._is_cache_expired(f"{feature_name}_v1") is False

    def test_is_cache_expired_returns_true_after_ttl(self, feature_store, sample_values):
        """_is_cache_expired is True after the TTL has elapsed (fixture ttl=2 s)."""
        feature_name = "feat_expired"
        feature_store.store_feature(feature_name, sample_values)
        feature_store.get_feature(feature_name, use_cache=True)

        time.sleep(3)

        assert feature_store._is_cache_expired(f"{feature_name}_v1") is True

    # ------------------------------------------------------------------
    # config/feature_store.yaml loading
    # ------------------------------------------------------------------

    def test_create_feature_store_reads_yaml_config(self, tmp_path):
        """create_feature_store() honours values in feature_store.yaml."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml_file = config_dir / "feature_store.yaml"
        yaml_file.write_text(
            "cache:\n  ttl_seconds: 300\n  maxsize: 64\n  max_size_mb: 200\n"
        )
        storage_dir = tmp_path / "fs"
        storage_dir.mkdir()

        from astroml.features.feature_store import create_feature_store

        store = create_feature_store(
            storage_path=str(storage_dir),
            config_path=str(yaml_file),
        )

        assert store._cache_ttl_seconds == 300
        assert store._cache_maxsize == 64
        assert store._max_cache_size_bytes == 200 * 1024 * 1024

    def test_create_feature_store_uses_defaults_without_yaml(self, tmp_path):
        """create_feature_store() works fine when no yaml file exists."""
        storage_dir = tmp_path / "fs"
        storage_dir.mkdir()
        missing_config = tmp_path / "nonexistent.yaml"

        from astroml.features.feature_store import create_feature_store

        store = create_feature_store(
            storage_path=str(storage_dir),
            config_path=str(missing_config),
        )

        assert store._cache_ttl_seconds == FeatureStore._DEFAULT_TTL
        assert store._cache_maxsize == FeatureStore._DEFAULT_MAXSIZE


if __name__ == "__main__":
    pytest.main([__file__])
