"""Tests for FeatureStore helper methods introduced in #508."""

from __future__ import annotations

import pytest

from astroml.features.feature_store import (
    FeatureDefinition,
    FeatureStatus,
    FeatureStorage,
    FeatureType,
    _safe_json_loads,
)


@pytest.fixture
def tmp_storage(tmp_path):
    return FeatureStorage(tmp_path / "store")


def test_safe_json_loads_valid():
    assert _safe_json_loads('{"a": 1}', {}) == {"a": 1}


def test_safe_json_loads_empty():
    assert _safe_json_loads("", 0) == 0
    assert _safe_json_loads(None, "x") == "x"


def test_safe_json_loads_invalid():
    assert _safe_json_loads("not json", []) == []


def test_row_to_dict_deserialization(tmp_storage):
    row = (
        "f_v1",
        "f",
        1,
        "desc",
        "numeric",
        '{"p": 1}',
        '["tag"]',
        "owner",
        "development",
        "2024-01-01T00:00:00",
        "2024-01-01T00:00:00",
        '{"m": 1}',
    )
    cols = [
        "feature_id",
        "name",
        "version",
        "description",
        "feature_type",
        "parameters",
        "tags",
        "owner",
        "status",
        "created_at",
        "updated_at",
        "metadata",
    ]
    data = FeatureStorage._row_to_dict(row, cols)
    assert data["parameters"] == {"p": 1}
    assert data["tags"] == ["tag"]
    assert data["metadata"] == {"m": 1}


def test_deserialize_feature_definition(tmp_storage):
    fd = FeatureDefinition(name="test", description="d", feature_type=FeatureType.NUMERIC)
    tmp_storage.store_feature_definition(fd)
    row = (
        tmp_storage.db_path.connect()
        .execute("SELECT * FROM feature_definitions WHERE feature_id = ?", (fd.feature_id,))
        .fetchone()
    )
    result = FeatureStorage._deserialize_feature_definition(row)
    assert result.name == "test"
    assert result.feature_type == FeatureType.NUMERIC


def test_get_feature_definition_returns_none(tmp_storage):
    assert tmp_storage.get_feature_definition("nonexistent") is None


def test_list_feature_definitions_empty(tmp_storage):
    assert tmp_storage.list_feature_definitions() == []


def test_list_feature_definitions_filters(tmp_storage):
    for i in range(3):
        fd = FeatureDefinition(
            name=f"feat{i}",
            description="d",
            feature_type=FeatureType.NUMERIC,
            status=FeatureStatus.PRODUCTION if i == 0 else FeatureStatus.DEVELOPMENT,
            tags=["common"] + (["a"] if i == 1 else ["b"]),
            owner="team",
        )
        tmp_storage.store_feature_definition(fd)

    prod = tmp_storage.list_feature_definitions(status=FeatureStatus.PRODUCTION)
    assert len(prod) == 1
    assert prod[0].name == "feat0"

    tagged = tmp_storage.list_feature_definitions(tags=["a"])
    assert len(tagged) == 1
    assert tagged[0].name == "feat1"

    owner = tmp_storage.list_feature_definitions(owner="team")
    assert len(owner) == 3
