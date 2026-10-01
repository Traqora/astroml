"""Tests for YAML-configurable feature builders (#742)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from astroml.features.feature_registry import FeatureRegistryService, FeatureType
from astroml.features.yaml_builders import (
    FEATURE_PRIMITIVES,
    BuilderSpecError,
    FeatureBuilderSpec,
    load_feature_builder_specs,
    load_feature_builders,
    register_feature_builder_specs,
)

VALID_YAML = """
builders:
  - name: tx_count_7d
    description: Count of transactions over 7 days
    primitive: count
    inputs:
      entity: account
      source: transactions
      columns: [timestamp]
    window: 7d
    output_column: tx_count_7d
    tags: [temporal]
  - name: avg_amount_7d
    description: Average amount over 7 days
    primitive: mean
    inputs:
      columns: [timestamp, amount]
    window: 7d
    output_column: avg_amount_7d
    params:
      value_column: amount
"""


def _write(tmp_path: Path, content: str) -> Path:
    config = tmp_path / "builders.yaml"
    config.write_text(content)
    return config


class TestWindowParsing:
    """Window string parsing and validation."""

    def test_parses_days_hours_minutes_seconds(self):
        spec = FeatureBuilderSpec(name="x", description="", primitive="count", window="7d")
        assert spec.window_seconds == 7 * 86400.0
        spec = FeatureBuilderSpec(name="x", description="", primitive="count", window="12h")
        assert spec.window_seconds == 12 * 3600.0
        spec = FeatureBuilderSpec(name="x", description="", primitive="count", window="30m")
        assert spec.window_seconds == 30 * 60.0
        spec = FeatureBuilderSpec(name="x", description="", primitive="count", window="45s")
        assert spec.window_seconds == 45.0

    def test_windowless_spec_has_none(self):
        spec = FeatureBuilderSpec(name="x", description="", primitive="count")
        assert spec.window is None
        assert spec.window_seconds is None

    @pytest.mark.parametrize("bad", ["7", "d7", "7x", "7dd", "", "  "])
    def test_invalid_window_rejected(self, bad):
        with pytest.raises(BuilderSpecError):
            FeatureBuilderSpec(name="x", description="", primitive="count", window=bad)

    def test_empty_name_rejected(self):
        with pytest.raises(BuilderSpecError):
            FeatureBuilderSpec(name="  ", description="", primitive="count")


class TestYamllLoading:
    """Loading specs from YAML files."""

    def test_loads_valid_file(self, tmp_path):
        specs = load_feature_builder_specs(_write(tmp_path, VALID_YAML))
        assert [s.name for s in specs] == ["tx_count_7d", "avg_amount_7d"]
        first = specs[0]
        assert first.primitive == "count"
        assert first.window == "7d"
        assert first.output_column == "tx_count_7d"
        assert first.entity == "account"
        assert first.source == "transactions"
        assert first.columns == ["timestamp"]
        assert first.tags == ["temporal"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(BuilderSpecError, match="not found"):
            load_feature_builder_specs(tmp_path / "nope.yaml")

    def test_missing_builders_key_raises(self, tmp_path):
        with pytest.raises(BuilderSpecError, match="builders"):
            load_feature_builder_specs(_write(tmp_path, "features: []"))

    def test_duplicate_names_rejected(self, tmp_path):
        entry = """uilders:
  - name: dup
    description: first
    primitive: count
  - name: dup
    description: second
    primitive: count
"""
        with pytest.raises(BuilderSpecError, match="duplicate"):
            load_feature_builder_specs(_write(tmp_path, entry))

    def test_unknown_keys_rejected(self, tmp_path):
        content = VALID_YAML.replace("tags: [temporal]", "tagz: [temporal]")
        with pytest.raises(BuilderSpecError, match="unknown keys"):
            load_feature_builder_specs(_write(tmp_path, content))

    def test_missing_primitive_rejected(self, tmp_path):
        content = """
builders:
  - name: no_primitive
    description: broken
"""
        with pytest.raises(BuilderSpecError, match="primitive"):
            load_feature_builder_specs(_write(tmp_path, content))

    def test_invalid_feature_type_rejected(self, tmp_path):
        content = VALID_YAML.replace(
            "primitive: count", "primitive: count\n    feature_type: banana"
        )
        with pytest.raises(BuilderSpecError, match="feature_type"):
            load_feature_builder_specs(_write(tmp_path, content))

    def test_broken_yaml_raises(self, tmp_path):
        with pytest.raises(BuilderSpecError, match="invalid YAML"):
            load_feature_builder_specs(_write(tmp_path, "builders: [ {oops"))

    def test_shipped_example_config_is_valid(self):
        """The example config in configs/ must always load cleanly."""
        repo_root = Path(__file__).resolve().parents[2]
        config = repo_root / "configs" / "feature_builders.yaml"
        if not config.exists():  # pragma: no cover - guards layout changes
            pytest.skip("example config not present")
        specs = load_feature_builder_specs(config)
        assert len(specs) >= 5


class TestPrimitives:
    """Primitive resolution and computation."""

    ROWS = [
        {"timestamp": "2026-01-01", "amount": 10.0, "counterparty": "a"},
        {"timestamp": "2026-01-02", "amount": 20.0, "counterparty": "b"},
        {"timestamp": "2026-01-03", "amount": 30.0, "counterparty": "a"},
    ]

    def _spec(self, primitive, **params):
        return FeatureBuilderSpec(name="x", description="", primitive=primitive, params=params)

    def test_unknown_primitive_rejected_at_load(self, tmp_path):
        with pytest.raises(BuilderSpecError, match="unknown primitive"):
            load_feature_builders(_write(tmp_path, VALID_YAML), primitives={})

    def test_count(self):
        assert FEATURE_PRIMITIVES["count"](self.ROWS, self._spec("count")) == 3.0

    def test_sum_mean_max_min(self):
        assert FEATURE_PRIMITIVES["sum"](self.ROWS, self._spec("sum")) == 60.0
        assert FEATURE_PRIMITIVES["mean"](self.ROWS, self._spec("mean")) == 20.0
        assert FEATURE_PRIMITIVES["max"](self.ROWS, self._spec("max")) == 30.0
        assert FEATURE_PRIMITIVES["min"](self.ROWS, self._spec("min")) == 10.0

    def test_std(self):
        assert FEATURE_PRIMITIVES["std"](self.ROWS[:1], self._spec("std")) == 0.0
        assert FEATURE_PRIMITIVES["std"](self.ROWS, self._spec("std")) > 0

    def test_unique_count(self):
        spec = self._spec("unique_count", value_column="counterparty")
        assert FEATURE_PRIMITIVES["unique_count"](self.ROWS, spec) == 2.0

    def test_ratio(self):
        rows = [{"in": 3.0, "out": 1.0}, {"in": 1.0, "out": 1.0}]
        spec = self._spec("ratio", numerator="in", denominator="out")
        assert FEATURE_PRIMITIVES["ratio"](rows, spec) == 2.0
        zero = self._spec("ratio", numerator="in", denominator="out")
        assert FEATURE_PRIMITIVES["ratio"]([{"in": 1.0, "out": 0.0}], zero) == 0.0

    def test_empty_rows(self):
        for name in ("count", "sum", "mean", "max", "min", "std"):
            assert FEATURE_PRIMITIVES[name]([], self._spec(name)) == 0.0

    def test_missing_column_raises(self):
        with pytest.raises(BuilderSpecError, match="missing"):
            FEATURE_PRIMITIVES["sum"](self.ROWS, self._spec("sum", value_column="nope"))


class TestDefinitionHash:
    """Definition hashing drives cache invalidation (#743)."""

    def test_same_definition_same_hash(self):
        a = FeatureBuilderSpec(name="x", description="", primitive="count", window="7d")
        b = FeatureBuilderSpec(name="x", description="", primitive="count", window="7d")
        assert a.definition_hash() == b.definition_hash()

    def test_window_change_changes_hash(self):
        a = FeatureBuilderSpec(name="x", description="", primitive="count", window="7d")
        b = FeatureBuilderSpec(name="x", description="", primitive="count", window="30d")
        assert a.definition_hash() != b.definition_hash()

    def test_param_change_changes_hash(self):
        a = FeatureBuilderSpec(name="x", description="", primitive="sum")
        b = FeatureBuilderSpec(
            name="x", description="", primitive="sum", params={"value_column": "fee"}
        )
        assert a.definition_hash() != b.definition_hash()


class TestRegistryIntegration:
    """Registration into FeatureRegistryService."""

    def test_register_specs(self, tmp_path):
        specs = load_feature_builder_specs(_write(tmp_path, VALID_YAML))
        registry = FeatureRegistryService()
        definitions = register_feature_builder_specs(registry, specs)

        assert len(definitions) == 2
        definition = registry.get_definition("tx_count_7d")
        assert definition is not None
        assert definition.feature_type == FeatureType.NUMERIC
        assert "yaml_builder" in definition.tags
        computer = registry.get_computer("tx_count_7d")
        assert computer is not None
        assert computer([{"timestamp": "t"}], entity_col="", timestamp_col="") == 1.0

    def test_register_with_custom_primitive(self, tmp_path):
        specs = load_feature_builder_specs(_write(tmp_path, VALID_YAML))
        registry = FeatureRegistryService()

        def double(rows, spec):
            return 2.0 * len(rows)

        # Custom primitives merge over the built-ins.
        merged = dict(FEATURE_PRIMITIVES)
        merged["count"] = double
        register_feature_builder_specs(registry, specs, primitives=merged)
        computer = registry.get_computer("tx_count_7d")
        assert computer([], entity_col="", timestamp_col="") == 0.0
        # The other builder still resolves its built-in primitive.
        assert registry.get_computer("avg_amount_7d") is not None

    def test_register_unknown_primitive_raises(self, tmp_path):
        specs = load_feature_builder_specs(_write(tmp_path, VALID_YAML))
        registry = FeatureRegistryService()
        with pytest.raises(BuilderSpecError):
            register_feature_builder_specs(registry, specs, primitives={})

    def test_registry_metadata_carries_definition_hash(self, tmp_path):
        specs = load_feature_builder_specs(_write(tmp_path, VALID_YAML))
        registry = FeatureRegistryService()
        (definition,) = [
            d for d in register_feature_builder_specs(registry, specs) if d.name == "tx_count_7d"
        ]
        assert definition.metadata["definition_hash"] == specs[0].definition_hash()


class TestShippedConfigPrimitives:
    """Every primitive referenced by the shipped config exists."""

    def test_shipped_config_primitives_resolve(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = repo_root / "configs" / "feature_builders.yaml"
        if not config.exists():  # pragma: no cover
            pytest.skip("example config not present")
        pairs = load_feature_builders(config)
        assert pairs, "shipped config should define builders"
        assert all(callable(fn) for _, fn in pairs)
