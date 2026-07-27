"""Tests for `load_database_config`'s validation + schema suggestions (#151)."""
from __future__ import annotations

import pathlib

import pytest

from astroml.db.session import load_database_config


def _write(path: pathlib.Path, content: str) -> pathlib.Path:
    path.write_text(content)
    return path


def test_empty_yaml_errors_with_schema_template(tmp_path: pathlib.Path) -> None:
    config = _write(tmp_path / "db.yaml", "")
    with pytest.raises(ValueError) as exc:
        load_database_config(config)
    msg = str(exc.value)
    assert "empty" in msg.lower()
    assert "database:" in msg
    assert "host:" in msg


def test_top_level_not_a_mapping_errors_with_schema_template(
    tmp_path: pathlib.Path,
) -> None:
    config = _write(tmp_path / "db.yaml", "- not a mapping\n- foo\n")
    with pytest.raises(ValueError) as exc:
        load_database_config(config)
    assert "must be a YAML mapping" in str(exc.value)


def test_missing_database_key_errors_with_schema_template(
    tmp_path: pathlib.Path,
) -> None:
    config = _write(tmp_path / "db.yaml", "other_root: 1\n")
    with pytest.raises(ValueError) as exc:
        load_database_config(config)
    msg = str(exc.value)
    assert "missing the `database:` key" in msg
    assert "host:" in msg


def test_database_value_must_be_mapping(tmp_path: pathlib.Path) -> None:
    config = _write(tmp_path / "db.yaml", "database: 5432\n")
    with pytest.raises(ValueError) as exc:
        load_database_config(config)
    assert "must be a mapping" in str(exc.value)


def test_invalid_port_errors_with_schema(tmp_path: pathlib.Path) -> None:
    config = _write(
        tmp_path / "db.yaml",
        "database:\n  host: localhost\n  port: 99999999\n  name: x\n  user: x\n",
    )
    with pytest.raises(ValueError) as exc:
        load_database_config(config)
    msg = str(exc.value)
    assert "Invalid database configuration" in msg
    assert "Expected schema" in msg


def test_valid_config_round_trips(tmp_path: pathlib.Path) -> None:
    config = _write(
        tmp_path / "db.yaml",
        "database:\n  host: db.example.com\n  port: 5432\n"
        "  name: astroml\n  user: astroml\n  password: secret\n",
    )
    cfg = load_database_config(config)
    assert cfg.host == "db.example.com"
    assert cfg.port == 5432
    assert cfg.to_url() == "postgresql://astroml:secret@db.example.com:5432/astroml"


def test_missing_file_raises_file_not_found(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_database_config(tmp_path / "does-not-exist.yaml")
