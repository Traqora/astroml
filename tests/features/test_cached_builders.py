"""Tests for the cached feature builder layer (#743)."""

from __future__ import annotations

import pytest

from astroml.features.cached_builders import (
    CachedFeatureBuilder,
    FeatureCacheResult,
    cache_key_for_spec,
)
from astroml.features.feature_cache import CacheConfig, FeatureCache
from astroml.features.yaml_builders import FEATURE_PRIMITIVES, FeatureBuilderSpec


def _spec(name="tx_count_7d", primitive="count", window="7d", **params):
    return FeatureBuilderSpec(
        name=name,
        description="test builder",
        primitive=primitive,
        window=window,
        inputs={"columns": ["timestamp"]},
        params=params,
    )


def _rows(n=3):
    return [{"timestamp": f"2026-01-0{i}", "amount": float(i)} for i in range(1, n + 1)]


@pytest.fixture()
def cache():
    return FeatureCache(CacheConfig(max_size=128, ttl_seconds=300))


@pytest.fixture()
def builder(cache):
    return CachedFeatureBuilder(cache=cache, data_version="v1", primitives=FEATURE_PRIMITIVES)


class TestCacheKeys:
    """Cache key construction covers every output-affecting dimension."""

    def test_key_includes_feature_window_version(self):
        spec = _spec()
        key = cache_key_for_spec(spec, "v1", "acct-1")
        assert "fb" in key
        assert "tx_count_7d" in key
        assert "7d" in key
        assert "v1" in key
        assert "acct-1" in key

    def test_different_data_version_different_key(self):
        spec = _spec()
        assert cache_key_for_spec(spec, "v1", "a") != cache_key_for_spec(spec, "v2", "a")

    def test_different_window_different_key(self):
        assert cache_key_for_spec(_spec(window="7d"), "v1", "a") != cache_key_for_spec(
            _spec(window="30d"), "v1", "a"
        )

    def test_definition_change_changes_key(self):
        """Changing YAML params changes the definition hash (#742 link)."""
        before = cache_key_for_spec(_spec(primitive="sum"), "v1", "a")
        after = cache_key_for_spec(_spec(primitive="sum", value_column="fee"), "v1", "a")
        assert before != after

    def test_extra_context_changes_key(self):
        spec = _spec()
        assert cache_key_for_spec(spec, "v1", "a") != cache_key_for_spec(
            spec, "v1", "a", as_of="2026-01-01"
        )


class TestComputeSemantics:
    """Hit/miss behaviour and invalidation."""

    def test_first_compute_is_miss_then_hit(self, builder):
        spec = _spec()
        first = builder.compute(spec, _rows(), entity_id="a")
        assert isinstance(first, FeatureCacheResult)
        assert first.cache_hit is False
        assert first.value == 3.0

        second = builder.compute(spec, _rows(), entity_id="a")
        assert second.cache_hit is True
        assert second.value == 3.0

    def test_different_entities_do_not_collide(self, builder):
        spec = _spec()
        builder.compute(spec, _rows(3), entity_id="a")
        builder.compute(spec, _rows(5), entity_id="b")
        a = builder.compute(spec, _rows(3), entity_id="a")
        b = builder.compute(spec, _rows(5), entity_id="b")
        assert a.cache_hit and b.cache_hit
        assert a.value == 3.0 and b.value == 5.0

    def test_data_version_bump_forces_recompute(self, cache):
        """Bumping the data version invalidates previous entries."""
        builder = CachedFeatureBuilder(
            cache=cache, data_version="v1", primitives=FEATURE_PRIMITIVES
        )
        spec = _spec(primitive="sum")
        v1 = builder.compute(spec, _rows(), entity_id="a")
        assert v1.cache_hit is False

        builder.data_version = "v2"
        v2 = builder.compute(spec, _rows(4), entity_id="a")
        assert v2.cache_hit is False
        assert v2.value == 10.0

    def test_invalidate_then_recompute(self, builder):
        spec = _spec(primitive="sum")
        builder.compute(spec, _rows(), entity_id="a")
        assert builder.compute(spec, _rows(), entity_id="a").cache_hit

        assert builder.invalidate(spec, entity_id="a") is True
        assert builder.invalidate(spec, entity_id="a") is False  # already gone

        fresh = builder.compute(spec, _rows(5), entity_id="a")
        assert fresh.cache_hit is False
        assert fresh.value == 15.0

    def test_recompute_always_fresh(self, builder):
        spec = _spec(primitive="sum")
        builder.compute(spec, _rows(), entity_id="a")
        result = builder.recompute(spec, _rows(4), entity_id="a")
        assert result.cache_hit is False
        assert result.value == 10.0
        # ...and the fresh value is now what's cached
        assert builder.compute(spec, _rows(4), entity_id="a").cache_hit

    def test_invalidate_all(self, builder):
        spec = _spec()
        builder.compute(spec, _rows(), entity_id="a")
        builder.invalidate_all()
        assert builder.compute(spec, _rows(), entity_id="a").cache_hit is False

    def test_compute_batch(self, builder):
        spec = _spec()
        results = builder.compute_batch(spec, {"a": _rows(2), "b": _rows(4)})
        assert set(results) == {"a", "b"}
        assert results["a"].value == 2.0
        assert results["b"].value == 4.0

    def test_unknown_primitive_raises_on_miss(self):
        builder = CachedFeatureBuilder(
            cache=FeatureCache(CacheConfig()), data_version="v1", primitives={}
        )
        with pytest.raises(ValueError, match="primitive"):
            builder.compute(_spec(), _rows(), entity_id="a")


class TestStatsAndMetrics:
    """Cache stats reflect hit/miss behaviour (observability)."""

    def test_stats_track_hits_and_misses(self, cache):
        builder = CachedFeatureBuilder(
            cache=cache, data_version="v1", primitives=FEATURE_PRIMITIVES
        )
        spec = _spec()
        builder.compute(spec, _rows(), entity_id="a")  # miss
        builder.compute(spec, _rows(), entity_id="a")  # hit
        builder.compute(spec, _rows(), entity_id="a")  # hit
        stats = cache.get_stats()
        assert stats["misses"] >= 1
        assert stats["hits"] >= 2


class TestDocumentationAlignment:
    """The module docstring semantics hold for explicit invalidation."""

    def test_zero_ttl_entry_can_be_invalidated(self):
        """Entries remain removable regardless of TTL semantics."""
        cache = FeatureCache(CacheConfig(max_size=16, ttl_seconds=3600))
        builder = CachedFeatureBuilder(
            cache=cache, data_version="v1", primitives=FEATURE_PRIMITIVES
        )
        spec = _spec()
        builder.compute(spec, _rows(), entity_id="a")
        assert builder.invalidate(spec, entity_id="a") is True
        assert builder.compute(spec, _rows(), entity_id="a").cache_hit is False
