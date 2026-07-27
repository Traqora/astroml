"""Tests for the top-level CLI help text and global flag wiring.

Regression coverage for #150 and #180 — the top-level help must surface
examples, the `--config` and `--env` flags, and the documented environment
variables, so new contributors can discover them from `--help` alone.
"""
from __future__ import annotations

import io
import os
import pathlib
from contextlib import redirect_stdout
from unittest import mock

import pytest

from astroml import cli


def _capture_help() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        cli.main(["--help"])
    return buf.getvalue()


def test_help_mentions_global_flags() -> None:
    output = _capture_help()
    assert "--config" in output
    assert "--env" in output


def test_help_includes_examples_section() -> None:
    output = _capture_help()
    assert "Examples:" in output
    assert "python -m astroml.cli" in output


def test_help_documents_env_vars() -> None:
    output = _capture_help()
    assert "ASTROML_DATABASE_URL" in output
    assert "ASTROML_ENV" in output


def test_env_flag_sets_astroml_env_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTROML_ENV", raising=False)
    fake_db = mock.Mock()
    fake_db.host = "localhost"
    fake_db.port = 5432
    fake_db.name = "x"
    fake_db.user = "u"
    fake_db.password = ""
    fake_db.to_url.return_value = "postgresql://u@localhost:5432/x"
    with mock.patch("astroml.cli.load_database_config", return_value=fake_db):
        with redirect_stdout(io.StringIO()):
            rc = cli.main(["--env", "production", "config", "--print-db"])
    assert rc == 0
    assert os.environ.get("ASTROML_ENV") == "production"


def test_env_flag_does_not_overwrite_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROML_ENV", "staging")
    fake_db = mock.Mock()
    fake_db.host = "h"
    fake_db.port = 5432
    fake_db.name = "n"
    fake_db.user = "u"
    fake_db.password = ""
    fake_db.to_url.return_value = "postgresql://u@h:5432/n"
    with mock.patch("astroml.cli.load_database_config", return_value=fake_db):
        with redirect_stdout(io.StringIO()):
            cli.main(["--env", "production", "config", "--print-db"])
    assert os.environ["ASTROML_ENV"] == "staging"


def test_config_flag_passes_path_to_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTROML_ENV", raising=False)
    fake_db = mock.Mock()
    fake_db.host = "h"
    fake_db.port = 5432
    fake_db.name = "n"
    fake_db.user = "u"
    fake_db.password = ""
    fake_db.to_url.return_value = "postgresql://u@h:5432/n"
    custom = pathlib.Path("custom/db.yaml")
    with mock.patch(
        "astroml.cli.load_database_config", return_value=fake_db
    ) as load_mock:
        with redirect_stdout(io.StringIO()):
            cli.main(["--config", str(custom), "config", "--print-db"])
    load_mock.assert_called_once_with(custom)


def test_help_lists_all_subcommands() -> None:
    """--help output must mention every top-level subcommand (#150)."""
    output = _capture_help()
    for subcommand in ("ingest", "config", "quickstart", "preprocess-backfill"):
        assert subcommand in output, f"subcommand {subcommand!r} missing from --help"


def test_help_mentions_readme_usage_link() -> None:
    """--help epilog must include a link to the README usage section (#150)."""
    output = _capture_help()
    assert "README" in output or "github.com" in output, (
        "--help should reference the README or project URL for further guidance"
    )


def test_quickstart_subcommand_help_mentions_key_flags() -> None:
    """quickstart --help must document --num-ledgers, --epochs, and --seed (#150)."""
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        cli.main(["quickstart", "--help"])
    output = buf.getvalue()
    for flag in ("--num-ledgers", "--epochs", "--seed"):
        assert flag in output, f"quickstart --help missing flag {flag!r}"
