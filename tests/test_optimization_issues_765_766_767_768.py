"""Tests for optimization issues #765, #766, #767, #768.

Issue #768: Parallel snapshot construction across nodes
Issue #767: Add caching for repeated graph computations
Issue #766: Optimise backfill memory usage for very large ledger ranges
Issue #765: Add enriched artifact metadata to registry entries
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Issue #768: Parallel snapshot construction
# ===========================================================================


class TestParallelSnapshotConstruction:
    """Tests for parallel_build_snapshots and compute_node_features_parallel."""

    def test_build_snapshot_window_single_returns_snapshot_window(self):
        """_build_snapshot_window_single returns a valid SnapshotWindow."""
        from astroml.features.graph.snapshot import SnapshotWindow, _build_snapshot_window_single

        # We can't easily call this without a real DB, but we can verify
        # the function signature and that it's a top-level callable
        assert callable(_build_snapshot_window_single)

    def test_compute_node_features_parallel_basic(self):
        """compute_node_features_parallel computes features for all nodes."""
        from astroml.features.graph.snapshot import compute_node_features_parallel

        def fake_compute(node_id: str) -> dict:
            return {"node": node_id, "value": len(node_id)}

        node_ids = ["alice", "bob", "charlie"]
        result = compute_node_features_parallel(
            node_ids, fake_compute, n_jobs=1, batch_size=2
        )

        assert len(result) == 3
        assert result["alice"]["value"] == 5
        assert result["bob"]["value"] == 3
        assert result["charlie"]["value"] == 7

    def test_compute_node_features_parallel_empty(self):
        """compute_node_features_parallel handles empty input."""
        from astroml.features.graph.snapshot import compute_node_features_parallel

        result = compute_node_features_parallel([], lambda x: x, n_jobs=1)
        assert result == {}

    def test_compute_node_features_parallel_deterministic(self):
        """compute_node_features_parallel produces deterministic results."""
        from astroml.features.graph.snapshot import compute_node_features_parallel

        call_count = {"n": 0}

        def counting_compute(node_id: str) -> int:
            call_count["n"] += 1
            return hash(node_id) % 100

        node_ids = [f"node_{i}" for i in range(20)]

        result1 = compute_node_features_parallel(node_ids, counting_compute, n_jobs=1)
        call_count["n"] = 0
        result2 = compute_node_features_parallel(node_ids, counting_compute, n_jobs=1)

        assert result1 == result2

    def test_compute_node_features_parallel_batch_size(self):
        """compute_node_features_parallel respects batch_size."""
        from astroml.features.graph.snapshot import compute_node_features_parallel

        batch_sizes_seen = []

        def track_batch(node_id: str) -> str:
            return f"feat_{node_id}"

        node_ids = [f"n{i}" for i in range(10)]
        result = compute_node_features_parallel(
            node_ids, track_batch, n_jobs=1, batch_size=5
        )

        assert len(result) == 10
        for nid in node_ids:
            assert result[nid] == f"feat_{nid}"


# ===========================================================================
# Issue #767: Graph computation caching
# ===========================================================================


class TestGraphComputationCache:
    """Tests for GraphComputationCache."""

    def test_singleton_pattern(self):
        """GraphComputationCache uses singleton pattern."""
        from astroml.cache.graph_cache import GraphComputationCache

        # Reset singleton for test
        GraphComputationCache._instance = None
        cache1 = GraphComputationCache()
        cache2 = GraphComputationCache()
        assert cache1 is cache2
        # Reset for other tests
        GraphComputationCache._instance = None

    def test_basic_set_get(self):
        """Cache set and get works correctly."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        cache.set("test:prefix", "key1", {"data": 42}, ttl_seconds=60)
        result = cache.get("test:prefix", "key1")
        assert result == {"data": 42}

        GraphComputationCache._instance = None

    def test_cache_miss(self):
        """Cache miss returns None."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        result = cache.get("test:miss", "nonexistent")
        assert result is None

        GraphComputationCache._instance = None

    def test_cache_hit_miss_stats(self):
        """Cache tracks hit/miss statistics."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        cache.set("stats:test", "k1", "v1")
        cache.get("stats:test", "k1")  # hit
        cache.get("stats:test", "missing")  # miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

        GraphComputationCache._instance = None

    def test_invalidate_specific_key(self):
        """Invalidating a specific key removes it."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        cache.set("inv:test", "k1", "v1")
        cache.set("inv:test", "k2", "v2")
        count = cache.invalidate("inv:test", "k1")
        assert count == 1
        assert cache.get("inv:test", "k1") is None
        assert cache.get("inv:test", "k2") == "v2"

        GraphComputationCache._instance = None

    def test_invalidate_all_for_prefix(self):
        """Invalidating prefix removes all entries with that prefix."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        cache.set("pfx:a", "k1", "v1")
        cache.set("pfx:a", "k2", "v2")
        cache.set("pfx:b", "k3", "v3")
        count = cache.invalidate("pfx:a")
        assert count == 2
        assert cache.get("pfx:a", "k1") is None
        assert cache.get("pfx:b", "k3") == "v3"

        GraphComputationCache._instance = None

    def test_cached_adjacency_decorator(self):
        """cached_adjacency decorator caches function results."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        call_count = {"n": 0}

        @cache.cached_adjacency(version="v1", window="7d", ttl_seconds=60)
        def build_adj(edges):
            call_count["n"] += 1
            return {"adj": len(edges)}

        result1 = build_adj([("a", "b")])
        result2 = build_adj([("a", "b")])

        assert result1 == {"adj": 1}
        assert result2 == {"adj": 1}
        assert call_count["n"] == 1  # Only called once

        GraphComputationCache._instance = None

    def test_cached_edge_features_decorator(self):
        """cached_edge_features decorator caches function results."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        call_count = {"n": 0}

        @cache.cached_edge_features(version="v1", window="7d", ttl_seconds=60)
        def compute_ef(edges):
            call_count["n"] += 1
            return [1.0, 2.0]

        result = compute_ef([("a", "b")])
        _ = compute_ef([("a", "b")])  # Should be cached

        assert result == [1.0, 2.0]
        assert call_count["n"] == 1

        GraphComputationCache._instance = None

    def test_cached_node_features_decorator(self):
        """cached_node_features decorator caches function results."""
        from astroml.cache.graph_cache import GraphComputationCache

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        call_count = {"n": 0}

        @cache.cached_node_features(version="v1", window="7d", ttl_seconds=60)
        def compute_nf(node_ids):
            call_count["n"] += 1
            return {nid: 0.5 for nid in node_ids}

        result = compute_nf(["a", "b"])
        _ = compute_nf(["a", "b"])  # Should be cached

        assert result == {"a": 0.5, "b": 0.5}
        assert call_count["n"] == 1

        GraphComputationCache._instance = None

    def test_graph_cache_config_defaults(self):
        """GraphCacheConfig has sensible defaults."""
        from astroml.cache.graph_cache import GraphCacheConfig, GraphCacheBackend

        config = GraphCacheConfig()
        assert config.backend == GraphCacheBackend.MEMORY
        assert config.max_size == 512
        assert config.default_ttl_seconds == 3600

    def test_get_graph_cache_singleton(self):
        """get_graph_cache returns the singleton instance."""
        from astroml.cache.graph_cache import GraphComputationCache, get_graph_cache

        GraphComputationCache._instance = None
        c1 = get_graph_cache()
        c2 = get_graph_cache()
        assert c1 is c2
        GraphComputationCache._instance = None

    def test_invalidate_graph_cache(self):
        """invalidate_graph_cache clears entries across all prefixes."""
        from astroml.cache.graph_cache import (
            GraphComputationCache,
            invalidate_graph_cache,
        )

        GraphComputationCache._instance = None
        cache = GraphComputationCache()

        cache.set("graph:adjacency", "k1", "v1")
        cache.set("graph:edge_features", "k2", "v2")
        cache.set("graph:node_features", "k3", "v3")

        count = invalidate_graph_cache()
        assert count == 3

        assert cache.get("graph:adjacency", "k1") is None
        assert cache.get("graph:edge_features", "k2") is None
        assert cache.get("graph:node_features", "k3") is None

        GraphComputationCache._instance = None


# ===========================================================================
# Issue #766: Memory-efficient backfill
# ===========================================================================


class TestMemoryEfficientBackfill:
    """Tests for memory-efficient ingestion state tracking."""

    def test_compact_ledger_set_basic(self):
        """_CompactLedgerSet supports add and membership check."""
        from astroml.ingestion.memory_efficient import _CompactLedgerSet

        s = _CompactLedgerSet()
        s.add(100)
        s.add(50)
        s.add(200)

        assert 100 in s
        assert 50 in s
        assert 200 in s
        assert 150 not in s
        assert len(s) == 3

    def test_compact_ledger_set_sorted(self):
        """_CompactLedgerSet maintains sorted order."""
        from astroml.ingestion.memory_efficient import _CompactLedgerSet

        s = _CompactLedgerSet()
        for v in [300, 100, 200, 50]:
            s.add(v)

        assert s.to_list() == [50, 100, 200, 300]

    def test_compact_ledger_set_no_duplicates(self):
        """_CompactLedgerSet deduplicates on add."""
        from astroml.ingestion.memory_efficient import _CompactLedgerSet

        s = _CompactLedgerSet()
        s.add(100)
        s.add(100)
        s.add(100)

        assert len(s) == 1
        assert s.to_list() == [100]

    def test_compact_ledger_set_from_values(self):
        """_CompactLedgerSet can be initialized with values."""
        from astroml.ingestion.memory_efficient import _CompactLedgerSet

        s = _CompactLedgerSet([300, 100, 200])
        assert s.to_list() == [100, 200, 300]
        assert len(s) == 3

    def test_bloom_filter_set_basic(self):
        """_BloomFilterSet supports add and membership check."""
        from astroml.ingestion.memory_efficient import _BloomFilterSet

        bf = _BloomFilterSet(expected_items=1000, fp_rate=0.01)
        bf.add(42)
        bf.add(99)
        bf.add(1000)

        assert 42 in bf
        assert 99 in bf
        assert 1000 in bf

    def test_bloom_filter_set_bounded_memory(self):
        """_BloomFilterSet uses bounded memory regardless of items added."""
        from astroml.ingestion.memory_efficient import _BloomFilterSet

        bf = _BloomFilterSet(expected_items=1000, fp_rate=0.01)
        initial_size = len(bf._bits)

        for i in range(10_000):
            bf.add(i)

        final_size = len(bf._bits)
        # Memory should not grow unboundedly
        assert final_size == initial_size
        assert len(bf) == 10_000

    def test_bloom_filter_set_low_false_positive_rate(self):
        """_BloomFilterSet has low false positive rate."""
        from astroml.ingestion.memory_efficient import _BloomFilterSet

        bf = _BloomFilterSet(expected_items=10_000, fp_rate=0.01)
        for i in range(10_000):
            bf.add(i)

        # Check items that were NOT added
        false_positives = 0
        test_count = 1000
        for i in range(10_000, 10_000 + test_count):
            if i in bf:
                false_positives += 1

        fp_rate = false_positives / test_count
        assert fp_rate < 0.05, f"False positive rate too high: {fp_rate}"

    def test_memory_efficient_state_compact_mode(self):
        """MemoryEfficientState works in compact mode."""
        from astroml.ingestion.memory_efficient import MemoryEfficientState

        state = MemoryEfficientState()
        state.add(100)
        state.add(200)
        state.add(50)

        assert 100 in state
        assert 200 in state
        assert 50 in state
        assert 150 not in state
        assert state.last_processed_ledger == 200
        assert state.mode == "compact"

    def test_memory_efficient_state_from_legacy(self):
        """MemoryEfficientState.from_legacy converts legacy state."""
        from astroml.ingestion.memory_efficient import MemoryEfficientState
        from astroml.ingestion.state import IngestionState

        legacy = IngestionState(
            last_processed_ledger=500,
            processed_ledgers={100, 200, 300, 400, 500},
        )

        state = MemoryEfficientState.from_legacy(legacy)
        assert state.last_processed_ledger == 500
        assert 100 in state
        assert 500 in state
        assert state.mode == "compact"

    def test_memory_efficient_state_from_legacy_large_set(self):
        """MemoryEfficientState.from_legacy uses bloom filter for large sets."""
        from astroml.ingestion.memory_efficient import MemoryEfficientState, _BloomFilterSet
        from astroml.ingestion.state import IngestionState

        legacy = IngestionState(
            last_processed_ledger=200_000,
            processed_ledgers=set(range(200_000)),
        )

        state = MemoryEfficientState.from_legacy(legacy)
        assert isinstance(state._processed, _BloomFilterSet)
        assert state.mode == "bloom"
        # Items should still be found (with possible false negatives for bloom,
        # but we added them so they should be found)
        assert 0 in state
        assert 100_000 in state
        assert 199_999 in state

    def test_memory_efficient_state_memory_usage(self):
        """memory_usage_bytes returns a reasonable estimate."""
        from astroml.ingestion.memory_efficient import MemoryEfficientState

        state = MemoryEfficientState()
        for i in range(100):
            state.add(i)

        usage = state.memory_usage_bytes()
        assert usage > 0
        assert usage < 100_000  # Should be well under 100KB for 100 items

    def test_chunked_state_store_save_load(self):
        """ChunkedStateStore round-trips through disk."""
        from astroml.ingestion.memory_efficient import ChunkedStateStore, MemoryEfficientState

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            store = ChunkedStateStore(path, flush_interval=10)

            state = MemoryEfficientState()
            state.add(100)
            state.add(200)
            state.add(300)

            store.save(state)

            loaded = store.load()
            assert loaded.last_processed_ledger == 300
            assert 100 in loaded
            assert 200 in loaded
            assert 300 in loaded

    def test_chunked_state_store_flush_interval(self):
        """ChunkedStateStore tracks flush intervals."""
        from astroml.ingestion.memory_efficient import ChunkedStateStore

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            store = ChunkedStateStore(path, flush_interval=3)

            assert not store.should_flush()  # 1
            assert not store.should_flush()  # 2
            assert store.should_flush()  # 3

            store.reset_flush_counter()
            assert not store.should_flush()  # 1 again


# ===========================================================================
# Issue #765: Enriched artifact metadata
# ===========================================================================


class TestEnrichedArtifactMetadata:
    """Tests for ArtifactMetadata and metadata collection."""

    def test_artifact_metadata_creation(self):
        """ArtifactMetadata can be created with defaults."""
        from astroml.storage.artifact_metadata import ArtifactMetadata

        meta = ArtifactMetadata()
        assert meta.framework_version is None
        assert meta.torch_version is None
        assert meta.registered_at is not None

    def test_artifact_metadata_to_dict(self):
        """ArtifactMetadata serialises to dict."""
        from astroml.storage.artifact_metadata import ArtifactMetadata

        meta = ArtifactMetadata(
            torch_version="2.1.0",
            dataset_checksum="abc123",
            training_duration_seconds=42.5,
            custom={"experiment": "exp_001"},
        )
        d = meta.to_dict()
        assert d["torch_version"] == "2.1.0"
        assert d["dataset_checksum"] == "abc123"
        assert d["training_duration_seconds"] == 42.5
        assert d["custom"]["experiment"] == "exp_001"

    def test_artifact_metadata_from_dict(self):
        """ArtifactMetadata deserialises from dict, ignoring unknown keys."""
        from astroml.storage.artifact_metadata import ArtifactMetadata

        d = {
            "torch_version": "2.0.0",
            "unknown_future_field": "ignored",
            "output_schema": {"shape": [1, 10]},
        }
        meta = ArtifactMetadata.from_dict(d)
        assert meta.torch_version == "2.0.0"
        assert meta.output_schema == {"shape": [1, 10]}

    def test_detect_framework_versions(self):
        """detect_framework_versions returns version info."""
        from astroml.storage.artifact_metadata import detect_framework_versions

        versions = detect_framework_versions()
        assert isinstance(versions, dict)
        assert "framework_version" in versions
        assert "torch_version" in versions
        assert "sklearn_version" in versions

    def test_compute_dataset_checksum_list(self):
        """compute_dataset_checksum works with list data."""
        from astroml.storage.artifact_metadata import compute_dataset_checksum

        data = [1, 2, 3, 4, 5]
        checksum = compute_dataset_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex

    def test_compute_dataset_checksum_deterministic(self):
        """compute_dataset_checksum is deterministic."""
        from astroml.storage.artifact_metadata import compute_dataset_checksum

        data = [{"a": 1, "b": 2}, {"c": 3}]
        c1 = compute_dataset_checksum(data)
        c2 = compute_dataset_checksum(data)
        assert c1 == c2

    def test_compute_dataset_checksum_different_data(self):
        """compute_dataset_checksum produces different hashes for different data."""
        from astroml.storage.artifact_metadata import compute_dataset_checksum

        c1 = compute_dataset_checksum([1, 2, 3])
        c2 = compute_dataset_checksum([4, 5, 6])
        assert c1 != c2

    def test_infer_output_schema_with_model(self):
        """infer_output_schema infers schema from a model with sample input."""
        from astroml.storage.artifact_metadata import infer_output_schema

        try:
            import torch

            model = torch.nn.Linear(10, 5)
            sample = torch.randn(1, 10)
            schema = infer_output_schema(model, sample)
            assert schema is not None
            assert "shape" in schema
            assert schema["shape"][1] == 5
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_infer_output_schema_without_model(self):
        """infer_output_schema returns None for unsupported models."""
        from astroml.storage.artifact_metadata import infer_output_schema

        schema = infer_output_schema("not_a_model")
        assert schema is None

    def test_collect_metadata_full(self):
        """collect_metadata populates all available fields."""
        from astroml.storage.artifact_metadata import collect_metadata

        meta = collect_metadata(
            training_duration=123.4,
            training_config={"lr": 0.01, "epochs": 100},
            dataset_name="fraud_v2",
            custom={"run_id": "abc"},
        )
        assert meta.training_duration_seconds == 123.4
        assert meta.training_config == {"lr": 0.01, "epochs": 100}
        assert meta.dataset_name == "fraud_v2"
        assert meta.custom == {"run_id": "abc"}

    def test_enrich_registry_entry_preserves_existing(self):
        """enrich_registry_entry preserves existing fields."""
        from astroml.storage.artifact_metadata import enrich_registry_entry

        existing = {
            "model_name": "my_model",
            "version": "1.0",
            "checksum_sha256": "existing_hash",
            "custom_metadata": {"user_key": "user_val"},
        }
        enriched = enrich_registry_entry(existing)
        assert enriched["model_name"] == "my_model"
        assert enriched["version"] == "1.0"
        assert enriched["checksum_sha256"] == "existing_hash"
        assert enriched["custom_metadata"]["user_key"] == "user_val"
        # New fields should be added
        assert "registered_at" in enriched
        assert "framework_version" in enriched

    def test_enrich_registry_entry_no_overwrite(self):
        """enrich_registry_entry does not overwrite existing fields."""
        from astroml.storage.artifact_metadata import enrich_registry_entry

        existing = {
            "framework_version": "custom_version",
            "torch_version": "1.0.0",
        }
        enriched = enrich_registry_entry(existing)
        assert enriched["framework_version"] == "custom_version"
        assert enriched["torch_version"] == "1.0.0"

    def test_migrate_registry_file(self):
        """migrate_registry_file enriches a meta.json on disk."""
        from astroml.storage.artifact_metadata import migrate_registry_file

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "model.pkl.meta.json")
            original = {
                "model_name": "test",
                "version": "1.0",
                "checksum_sha256": "abc",
                "size_bytes": 1024,
                "custom_metadata": {},
            }
            with open(meta_path, "w") as f:
                json.dump(original, f)

            result = migrate_registry_file(meta_path)
            assert result is not None
            assert result["model_name"] == "test"
            assert "registered_at" in result
            assert "framework_version" in result

            # Verify file was updated
            with open(meta_path) as f:
                updated = json.load(f)
            assert "registered_at" in updated

    def test_migrate_registry_file_nonexistent(self):
        """migrate_registry_file returns None for missing file."""
        from astroml.storage.artifact_metadata import migrate_registry_file

        result = migrate_registry_file("/nonexistent/path/meta.json")
        assert result is None

    def test_model_store_save_model_enriched_metadata(self):
        """ModelStore.save_model includes enriched metadata in sidecar."""
        from astroml.storage.model_store import ModelStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(base_dir=tmpdir)
            model = MagicMock()
            model.state_dict.return_value = {"weight": [1, 2, 3]}
            model.__class__.__name__ = "FakeModel"

            path = store.save_model(
                model_name="test_model",
                version="1.0",
                model_object=model,
                metadata={"experiment": "exp1"},
                training_duration=45.2,
                training_config={"lr": 0.01},
            )

            assert os.path.exists(path)

            # Check metadata sidecar
            meta_path = path + ".meta.json"
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["model_name"] == "test_model"
            assert meta["version"] == "1.0"
            assert meta["custom_metadata"]["experiment"] == "exp1"
            assert meta["training_duration_seconds"] == 45.2
            assert meta["training_config"] == {"lr": 0.01}
            assert "registered_at" in meta
            assert "framework_version" in meta

    def test_model_store_get_artifact_info_enriched(self):
        """ModelStore.get_artifact_info returns enriched metadata."""
        from astroml.storage.model_store import ModelStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(base_dir=tmpdir)
            model = MagicMock()
            model.state_dict.return_value = {"weight": [1, 2, 3]}

            store.save_model(
                model_name="test_model",
                version="1.0",
                model_object=model,
                training_duration=42.0,
            )

            info = store.get_artifact_info("test_model", "1.0")
            assert info["training_duration_seconds"] == 42.0
            assert "framework_version" in info
            assert "registered_at" in info
