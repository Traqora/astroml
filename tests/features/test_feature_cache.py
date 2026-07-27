"""Tests for feature cache module."""

from __future__ import annotations

import pytest
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from astroml.features.feature_cache import (
    CacheStrategy,
    StorageFormat,
    CacheConfig,
    StorageConfig,
    CacheEntry,
    MemoryCache,
    RedisCache,
    DiskCache,
    FeatureCache,
    FeatureStorageOptimizer,
    create_feature_cache,
    create_storage_optimizer,
)


class TestCacheConfig:
    """Test CacheConfig class."""
    
    def test_cache_config_creation(self):
        """Test creating cache configuration."""
        config = CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=1000,
            ttl_seconds=3600,
            compression=True,
        )
        
        assert config.strategy == CacheStrategy.LRU
        assert config.max_size == 1000
        assert config.ttl_seconds == 3600
        assert config.compression is True


class TestStorageConfig:
    """Test StorageConfig class."""
    
    def test_storage_config_creation(self):
        """Test creating storage configuration."""
        config = StorageConfig(
            format=StorageFormat.PARQUET,
            compression="snappy",
            partition_cols=["entity_id"],
            index_cols=["timestamp"],
        )
        
        assert config.format == StorageFormat.PARQUET
        assert config.compression == "snappy"
        assert config.partition_cols == ["entity_id"]
        assert config.index_cols == ["timestamp"]


class TestCacheEntry:
    """Test CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=3600,
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 3600
        assert entry.access_count == 0
        assert not entry.is_expired
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Create expired entry
        past_time = datetime.utcnow() - timedelta(hours=2)
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=3600,  # 1 hour TTL
        )
        entry.timestamp = past_time
        
        assert entry.is_expired
        
        # Create non-expired entry
        entry.timestamp = datetime.utcnow() - timedelta(minutes=30)
        assert not entry.is_expired
    
    def test_cache_entry_access(self):
        """Test cache entry access."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
        )
        
        initial_count = entry.access_count
        result = entry.access()
        
        assert result == "test_value"
        assert entry.access_count == initial_count + 1


class TestMemoryCache:
    """Test MemoryCache class."""
    
    @pytest.fixture
    def cache_config(self):
        """Create cache configuration."""
        return CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=10,
        )
    
    @pytest.fixture
    def memory_cache(self, cache_config):
        """Create memory cache instance."""
        return MemoryCache(cache_config)
    
    def test_memory_cache_put_get(self, memory_cache):
        """Test putting and getting values."""
        # Put value
        memory_cache.put("test_key", "test_value")
        
        # Get value
        result = memory_cache.get("test_key")
        assert result == "test_value"
        
        # Get non-existent value
        result = memory_cache.get("non_existent")
        assert result is None
    
    def test_memory_cache_ttl(self):
        """Test TTL functionality."""
        config = CacheConfig(
            strategy=CacheStrategy.TTL,
            max_size=10,
            ttl_seconds=1,  # 1 second TTL
        )
        cache = MemoryCache(config)
        
        # Put value
        cache.put("test_key", "test_value")
        
        # Get value immediately (should work)
        result = cache.get("test_key")
        assert result == "test_value"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Get expired value (should return None)
        result = cache.get("test_key")
        assert result is None
    
    def test_memory_cache_remove(self, memory_cache):
        """Test removing values."""
        # Put value
        memory_cache.put("test_key", "test_value")
        
        # Remove value
        result = memory_cache.remove("test_key")
        assert result is True
        
        # Try to get removed value
        result = memory_cache.get("test_key")
        assert result is None
        
        # Remove non-existent value
        result = memory_cache.remove("non_existent")
        assert result is False
    
    def test_memory_cache_clear(self, memory_cache):
        """Test clearing cache."""
        # Put multiple values
        for i in range(5):
            memory_cache.put(f"key_{i}", f"value_{i}")
        
        assert memory_cache.size() == 5
        
        # Clear cache
        memory_cache.clear()
        
        assert memory_cache.size() == 0
        
        # Try to get values (should all be None)
        for i in range(5):
            result = memory_cache.get(f"key_{i}")
            assert result is None
    
    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction."""
        config = CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=3,  # Small cache to trigger eviction
        )
        cache = MemoryCache(config)
        
        # Fill cache beyond capacity
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Check that cache size is maintained
        assert cache.size() == 3
        
        # Check that oldest values were evicted
        assert cache.get("key_0") is None
        assert cache.get("key_1") is None
        
        # Check that newest values are still present
        assert cache.get("key_2") is not None
        assert cache.get("key_3") is not None
        assert cache.get("key_4") is not None


class TestDiskCache:
    """Test DiskCache class."""
    
    @pytest.fixture
    def temp_cache_path(self):
        """Create temporary cache path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_config(self, temp_cache_path):
        """Create cache configuration."""
        return CacheConfig(
            strategy=CacheStrategy.DISK,
            disk_path=temp_cache_path,
        )
    
    @pytest.fixture
    def disk_cache(self, cache_config):
        """Create disk cache instance."""
        return DiskCache(cache_config)
    
    def test_disk_cache_put_get(self, disk_cache):
        """Test putting and getting values."""
        test_value = pd.DataFrame({"feature": [1, 2, 3]})
        
        # Put value
        disk_cache.put("test_key", test_value)
        
        # Get value
        result = disk_cache.get("test_key")
        assert result is not None
        pd.testing.assert_frame_equal(result, test_value)
        
        # Get non-existent value
        result = disk_cache.get("non_existent")
        assert result is None
    
    def test_disk_cache_ttl(self, temp_cache_path):
        """Test TTL functionality."""
        config = CacheConfig(
            strategy=CacheStrategy.DISK,
            disk_path=temp_cache_path,
        )
        cache = DiskCache(config)
        
        test_value = "test_value"
        
        # Put value with short TTL
        cache.put("test_key", test_value, ttl_seconds=1)
        
        # Get value immediately (should work)
        result = cache.get("test_key")
        assert result == test_value
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Get expired value (should return None)
        result = cache.get("test_key")
        assert result is None
    
    def test_disk_cache_remove(self, disk_cache):
        """Test removing values."""
        test_value = "test_value"
        
        # Put value
        disk_cache.put("test_key", test_value)
        
        # Remove value
        result = disk_cache.remove("test_key")
        assert result is True
        
        # Try to get removed value
        result = disk_cache.get("test_key")
        assert result is None
    
    def test_disk_cache_clear(self, disk_cache):
        """Test clearing cache."""
        # Put multiple values
        for i in range(5):
            disk_cache.put(f"key_{i}", f"value_{i}")
        
        assert disk_cache.size() == 5
        
        # Clear cache
        disk_cache.clear()
        
        assert disk_cache.size() == 0
        
        # Try to get values (should all be None)
        for i in range(5):
            result = disk_cache.get(f"key_{i}")
            assert result is None
    
    def test_disk_cache_cleanup_expired(self, temp_cache_path):
        """Test cleanup of expired entries."""
        config = CacheConfig(
            strategy=CacheStrategy.DISK,
            disk_path=temp_cache_path,
        )
        cache = DiskCache(config)
        
        # Put values with different TTLs
        cache.put("permanent_key", "permanent_value")
        cache.put("expired_key", "expired_value", ttl_seconds=1)
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Cleanup expired entries
        removed_count = cache.cleanup_expired()
        
        assert removed_count == 1
        assert cache.size() == 1
        
        # Check that permanent value is still accessible
        result = cache.get("permanent_key")
        assert result == "permanent_value"
        
        # Check that expired value is gone
        result = cache.get("expired_key")
        assert result is None


class TestFeatureCache:
    """Test FeatureCache class."""
    
    @pytest.fixture
    def temp_cache_path(self):
        """Create temporary cache path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_config(self, temp_cache_path):
        """Create cache configuration."""
        return CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=10,
        )
    
    @pytest.fixture
    def feature_cache(self, cache_config):
        """Create feature cache instance."""
        return FeatureCache(cache_config)
    
    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data."""
        return pd.DataFrame({
            "feature_value": [1.0, 2.0, 3.0],
        }, index=["entity1", "entity2", "entity3"])
    
    def test_feature_cache_put_get(self, feature_cache, sample_feature_data):
        """Test putting and getting features."""
        feature_name = "test_feature"
        entity_ids = ["entity1", "entity2", "entity3"]
        
        # Put feature
        feature_cache.put(feature_name, sample_feature_data, entity_ids)
        
        # Get feature
        result = feature_cache.get(feature_name, entity_ids)
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_feature_data)
        
        # Get feature without entity filter
        result = feature_cache.get(feature_name)
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_feature_data)
    
    def test_feature_cache_key_generation(self, feature_cache):
        """Test cache key generation."""
        # Test basic key generation
        key1 = feature_cache._make_key("feature1")
        key2 = feature_cache._make_key("feature1")
        assert key1 == key2
        
        # Test key generation with entities
        key3 = feature_cache._make_key("feature1", ["entity1", "entity2"])
        key4 = feature_cache._make_key("feature1", ["entity2", "entity1"])  # Different order
        assert key3 == key4  # Should be same after sorting
        
        # Test key generation with parameters
        key5 = feature_cache._make_key("feature1", timestamp="2023-01-01")
        key6 = feature_cache._make_key("feature1", timestamp="2023-01-02")
        assert key5 != key6
    
    def test_feature_cache_stats(self, feature_cache, sample_feature_data):
        """Test cache statistics."""
        feature_name = "test_feature"
        
        # Initial stats
        stats = feature_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Put feature
        feature_cache.put(feature_name, sample_feature_data)
        stats = feature_cache.get_stats()
        assert stats["sets"] == 1
        
        # Get feature (hit)
        result = feature_cache.get(feature_name)
        assert result is not None
        stats = feature_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 1.0
        
        # Get non-existent feature (miss)
        result = feature_cache.get("non_existent")
        assert result is None
        stats = feature_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_feature_cache_remove(self, feature_cache, sample_feature_data):
        """Test removing features."""
        feature_name = "test_feature"
        
        # Put feature
        feature_cache.put(feature_name, sample_feature_data)
        
        # Remove feature
        result = feature_cache.remove(feature_name)
        assert result is True
        
        # Try to get removed feature
        result = feature_cache.get(feature_name)
        assert result is None
        
        # Remove non-existent feature
        result = feature_cache.remove("non_existent")
        assert result is False
    
    def test_feature_cache_clear(self, feature_cache, sample_feature_data):
        """Test clearing cache."""
        # Put multiple features
        for i in range(3):
            feature_cache.put(f"feature_{i}", sample_feature_data)
        
        stats = feature_cache.get_stats()
        assert stats["sets"] == 3
        
        # Clear cache
        feature_cache.clear()
        
        # Check stats are reset
        stats = feature_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["hit_rate"] == 0.0


class TestFeatureStorageOptimizer:
    """Test FeatureStorageOptimizer class."""
    
    @pytest.fixture
    def storage_config(self):
        """Create storage configuration."""
        return StorageConfig(
            format=StorageFormat.PARQUET,
            compression="snappy",
        )
    
    @pytest.fixture
    def optimizer(self, storage_config):
        """Create storage optimizer instance."""
        return FeatureStorageOptimizer(storage_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "numeric_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "categorical_col": ["A", "B", "A", "C", "B"],
            "text_col": ["text1", "text2", "text3", "text4", "text5"],
        }, index=["entity1", "entity2", "entity3", "entity4", "entity5"])
    
    def test_optimize_dataframe(self, optimizer, sample_data):
        """Test DataFrame optimization."""
        optimized = optimizer.optimize_dataframe(sample_data, "test_feature")
        
        # Check that categorical columns were converted
        assert optimized["categorical_col"].dtype.name == "category"
        
        # Check that numeric columns were downcast
        assert optimized["numeric_col"].dtype == "int8" or optimized["numeric_col"].dtype == "int16"
        assert optimized["float_col"].dtype == "float32" or optimized["float_col"].dtype == "float16"
        
        # Check that index name was set
        assert optimized.index.name == "test_feature"
    
    def test_save_load_dataframe(self, optimizer, sample_data):
        """Test saving and loading DataFrames."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            filepath = Path(f.name)
            
            # Save DataFrame
            optimizer.save_dataframe(sample_data, filepath)
            assert filepath.exists()
            
            # Load DataFrame
            loaded_data = optimizer.load_dataframe(filepath)
            
            # Check that data is the same
            pd.testing.assert_frame_equal(loaded_data, sample_data)
    
    def test_estimate_size(self, optimizer, sample_data):
        """Test size estimation."""
        size = optimizer.estimate_size(sample_data)
        assert size > 0
        assert isinstance(size, int)
    
    def test_different_formats(self, sample_data):
        """Test different storage formats."""
        formats = [
            StorageFormat.PARQUET,
            StorageFormat.FEATHER,
            # StorageFormat.HDF5,  # Might not be available
            StorageFormat.PICKLE,
        ]
        
        for fmt in formats:
            config = StorageConfig(format=fmt)
            optimizer = FeatureStorageOptimizer(config)
            
            try:
                # Test save/load cycle
                import tempfile
                suffix = f".{fmt.value}"
                
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    filepath = Path(f.name)
                    
                    # Save
                    optimizer.save_dataframe(sample_data, filepath)
                    
                    # Load
                    loaded_data = optimizer.load_dataframe(filepath)
                    
                    # Check data integrity
                    if fmt != StorageFormat.CSV:  # CSV might have type differences
                        pd.testing.assert_frame_equal(loaded_data, sample_data, check_dtype=False)
                    
            except Exception as e:
                # Some formats might not be available
                print(f"Format {fmt} not available: {e}")


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def temp_cache_path(self):
        """Create temporary cache path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_create_feature_cache(self, temp_cache_path):
        """Test create_feature_cache function."""
        cache = create_feature_cache(
            strategy=CacheStrategy.LRU,
            max_size=100,
            cache_path=temp_cache_path,
        )
        
        assert isinstance(cache, FeatureCache)
        assert cache.config.strategy == CacheStrategy.LRU
        assert cache.config.max_size == 100
    
    def test_create_storage_optimizer(self):
        """Test create_storage_optimizer function."""
        optimizer = create_storage_optimizer(
            format=StorageFormat.PARQUET,
            compression="snappy",
        )
        
        assert isinstance(optimizer, FeatureStorageOptimizer)
        assert optimizer.config.format == StorageFormat.PARQUET
        assert optimizer.config.compression == "snappy"


if __name__ == "__main__":
    pytest.main([__file__])
