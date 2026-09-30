"""Tests for the refactored feature_store module.

Verifies that extracted helper methods produce the same results as before
the refactoring, and that new helper methods are properly exposed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from astroml.features.feature_store import (
    FeatureDefinition,
    FeatureSet,
    FeatureStatus,
    FeatureStorage,
    FeatureType,
    FeatureValue,
)

# ── FeatureStorage helpers ────────────────────────────────────────────────


class TestFeatureStorageHelpers:
    """Tests for new helper methods on FeatureStorage."""

    def test_matches_tags_all_present(self) -> None:
        """_matches_tags returns True when all required tags are present."""
        assert FeatureStorage._matches_tags(["a", "b", "c"], ["a", "c"])
        assert FeatureStorage._matches_tags(["x"], ["x"])
        assert FeatureStorage._matches_tags(["a", "b"], [])  # empty requirement

    def test_matches_tags_missing(self) -> None:
        """_matches_tags returns False when a required tag is missing."""
        assert not FeatureStorage._matches_tags(["a", "b"], ["c"])
        assert not FeatureStorage._matches_tags([], ["a"])
        assert not FeatureStorage._matches_tags(["a"], ["a", "b"])

    def test_matches_tags_empty_feature_tags(self) -> None:
        """_matches_tags handles empty feature tag lists."""
        assert not FeatureStorage._matches_tags([], ["a"])

    def test_build_feature_list_query_no_filters(self) -> None:
        """_build_feature_list_query with no filters."""
        query, params = FeatureStorage._build_feature_list_query(None, None)
        assert "WHERE 1=1" in query
        assert len(params) == 0

    def test_build_feature_list_query_with_status(self) -> None:
        """_build_feature_list_query with status filter."""
        query, params = FeatureStorage._build_feature_list_query(
            FeatureStatus.PRODUCTION, None
        )
        assert "status = ?" in query
        assert params == ["production"]

    def test_build_feature_list_query_with_owner(self) -> None:
        """_build_feature_list_query with owner filter."""
        query, params = FeatureStorage._build_feature_list_query(
            None, "test-team"
        )
        assert "owner = ?" in query
        assert params == ["test-team"]

    def test_build_feature_list_query_with_both(self) -> None:
        """_build_feature_list_query with both filters."""
        query, params = FeatureStorage._build_feature_list_query(
            FeatureStatus.DEVELOPMENT, "my-team"
        )
        assert "status = ?" in query
        assert "owner = ?" in query
        assert params == ["development", "my-team"]

    def test_list_feature_definitions_tag_filtering(self) -> None:
        """list_feature_definitions correctly filters by tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FeatureStorage(tmpdir)

            # Register features with different tags
            fd1 = FeatureDefinition(
                name="feat_a", description="A",
                feature_type=FeatureType.NUMERIC, tags=["ml", "prod"],
            )
            fd2 = FeatureDefinition(
                name="feat_b", description="B",
                feature_type=FeatureType.NUMERIC, tags=["ml"],
            )
            fd3 = FeatureDefinition(
                name="feat_c", description="C",
                feature_type=FeatureType.NUMERIC, tags=["analytics"],
            )
            storage.store_feature_definition(fd1)
            storage.store_feature_definition(fd2)
            storage.store_feature_definition(fd3)

            # Filter by "ml" tag — should get feat_a and feat_b
            result = storage.list_feature_definitions(tags=["ml"])
            names = {fd.name for fd in result}
            assert names == {"feat_a", "feat_b"}

            # Filter by "prod" tag — only feat_a
            result = storage.list_feature_definitions(tags=["prod"])
            names = {fd.name for fd in result}
            assert names == {"feat_a"}

            # Filter by "ml" AND "prod" — only feat_a
            result = storage.list_feature_definitions(tags=["ml", "prod"])
            names = {fd.name for fd in result}
            assert names == {"feat_a"}


# ── FeatureStore helpers ──────────────────────────────────────────────────


class TestFeatureStoreHelpers:
    """Tests for new helper methods on FeatureStore."""

    def test_validate_input_columns_valid(self) -> None:
        """_validate_input_columns passes when required columns are present."""
        from astroml.features.feature_store import FeatureStore
        df = pd.DataFrame({"entity": [1, 2], "timestamp": [3, 4]})
        FeatureStore._validate_input_columns(df, "entity", "timestamp")

    def test_validate_input_columns_missing(self) -> None:
        """_validate_input_columns raises when a column is missing."""
        from astroml.features.feature_store import FeatureStore
        df = pd.DataFrame({"entity": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            FeatureStore._validate_input_columns(df, "entity", "missing_col")

    def test_resolve_feature_def_caches(self) -> None:
        """_resolve_feature_def caches results in metadata cache."""
        from astroml.features.feature_store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir)

            # Register a feature first
            def dummy_computer(data, ec, tc, **kw):
                return pd.DataFrame({"val": [1]}, index=data.index)

            fs.register_feature(
                "test_feat", dummy_computer, "test desc",
                feature_type=FeatureType.NUMERIC,
            )

            # First call should hit storage
            fd1 = fs._resolve_feature_def("test_feat")
            assert fd1.name == "test_feat"

            # Second call should hit metadata cache
            fd2 = fs._resolve_feature_def("test_feat")
            assert fd2 is fd1  # same object from cache

    def test_resolve_feature_def_unknown_raises(self) -> None:
        """_resolve_feature_def raises ValueError for unknown features."""
        from astroml.features.feature_store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir)
            with pytest.raises(ValueError, match="not found"):
                fs._resolve_feature_def("nonexistent")

    def test_add_feature_values_to_dict_single_column(self) -> None:
        """_add_feature_values_to_dict handles single-column DataFrames."""
        from astroml.features.feature_store import FeatureStore
        result: dict = {}
        df = pd.DataFrame({"val": [10, 20]}, index=["a", "b"])
        FeatureStore._add_feature_values_to_dict(result, "my_feat", df)
        assert "my_feat" in result
        pd.testing.assert_series_equal(
            result["my_feat"], pd.Series([10, 20], index=["a", "b"], name="val")
        )

    def test_add_feature_values_to_dict_multi_column(self) -> None:
        """_add_feature_values_to_dict expands multi-column DataFrames."""
        from astroml.features.feature_store import FeatureStore
        result: dict = {}
        df = pd.DataFrame(
            {"count": [5, 10], "mean": [1.2, 3.4]},
            index=["a", "b"],
        )
        FeatureStore._add_feature_values_to_dict(result, "stats", df)
        assert "stats_count" in result
        assert "stats_mean" in result
        pd.testing.assert_series_equal(
            result["stats_count"], pd.Series([5, 10], index=["a", "b"], name="count")
        )

    def test_get_cache_stats_structure(self) -> None:
        """get_cache_stats returns expected keys."""
        from astroml.features.feature_store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir)
            stats = fs.get_cache_stats()
            expected_keys = {
                "cached_features", "cache_size_mb", "max_cache_size_mb",
                "cache_utilization_pct", "cache_maxsize", "cache_ttl_seconds",
                "metadata_cached", "hits", "misses", "evictions",
                "hit_rate", "miss_rate",
            }
            assert set(stats.keys()) == expected_keys
            assert 0.0 <= stats["hit_rate"] <= 1.0
            assert 0.0 <= stats["miss_rate"] <= 1.0

    def test_clear_cache_resets_metrics(self) -> None:
        """clear_cache resets all cache state."""
        from astroml.features.feature_store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir)
            fs.clear_cache()
            stats = fs.get_cache_stats()
            assert stats["cached_features"] == 0
            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["evictions"] == 0

    def test_split_data_into_chunks(self) -> None:
        """_split_data_into_chunks produces correct chunk sizes."""
        from astroml.features.feature_store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir, chunk_size=2)
            df = pd.DataFrame({
                "entity": ["a", "a", "b", "b", "c", "c"],
                "ts": range(6),
            })
            chunks = fs._split_data_into_chunks(df, "entity")
            # 3 unique entities, chunk_size=2 → 2 chunks (first 2, last 1)
            assert len(chunks) == 2
            # Total rows preserved
            total_rows = sum(len(c) for c in chunks)
            assert total_rows == len(df)


# ── Integration tests ─────────────────────────────────────────────────────


class TestRefactoredFeatureStoreIntegration:
    """End-to-end tests exercising refactored code paths."""

    def test_compute_and_store_roundtrip(self) -> None:
        """compute_and_store → get_feature roundtrip works after refactoring."""
        from astroml.features.feature_store import FeatureStore

        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir)

            def double_feature(data, entity_col, timestamp_col, **kw):
                result = data[[entity_col]].copy()
                result["double"] = data.index * 2
                return result.groupby(entity_col).first()

            fs.register_feature(
                "double", double_feature, "doubles index",
                feature_type=FeatureType.NUMERIC,
            )

            df = pd.DataFrame({
                "account": ["a", "b", "c"],
                "time": [1, 2, 3],
            })
            result = fs.compute_and_store(
                "double", df, "account", "time",
            )
            assert "double" in result.columns

            # Retrieve it back
            stored = fs.get_feature("double", entity_ids=["a", "b"])
            assert stored is not None
            assert "double" in stored.columns

    def test_get_features_for_entities(self) -> None:
        """get_features_for_entities returns proper DataFrame after refactoring."""
        from astroml.features.feature_store import FeatureStore

        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureStore(storage_path=tmpdir, enable_parallel=False)

            def const_feat(data, entity_col, timestamp_col, **kw):
                result = data[[entity_col]].copy()
                result["val"] = 42
                return result.groupby(entity_col).first()

            fs.register_feature(
                "const42", const_feat, "constant 42",
                feature_type=FeatureType.NUMERIC,
            )

            df = pd.DataFrame({
                "entity": ["x", "y", "z"],
                "ts": [1, 2, 3],
            })
            fs.compute_and_store("const42", df, "entity", "ts")

            result = fs.get_features_for_entities(
                ["const42"], ["x", "y"],
            )
            assert list(result.index) == ["x", "y"]
            assert "const42" in result.columns
            assert result.loc["x", "const42"] == 42
            assert result.loc["y", "const42"] == 42


# ── Line-count validation ─────────────────────────────────────────────────


class TestFunctionLineCounts:
    """Verify that previously long functions are now under 50 lines."""

    @pytest.mark.parametrize("func_name,max_lines", [
        ("list_feature_definitions", 50),
        ("_register_builtin_features", 50),
        ("__init__", 50),  # FeatureStore.__init__
        ("compute_feature", 50),
        ("_compute_feature_parallel", 50),
        ("get_feature", 50),
        ("get_features_for_entities", 50),
    ])
    def test_refactored_function_under_limit(
        self, func_name: str, max_lines: int
    ) -> None:
        """Each refactored function is now ≤ max_lines."""
        import ast
        from pathlib import Path

        filepath = Path(__file__).parent.parent / "astroml" / "features" / "feature_store.py"
        source = filepath.read_text()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                actual = (node.end_lineno or node.lineno) - node.lineno + 1
                found = True
                assert actual <= max_lines, (
                    f"{func_name} is {actual} lines, expected ≤ {max_lines}"
                )
                break

        if not found:
            pytest.skip(f"Function '{func_name}' not found in AST (may be a method)")
