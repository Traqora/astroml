"""CLI help snapshot tests for the normalizer (issue #990).

To update after an intentional CLI change, regenerate
``tests/snapshots/normalizer_cli_help.txt`` with ``COLUMNS=80``.
"""

import pathlib

import pytest

from astroml.ingestion import normalizer

SNAPSHOT = pathlib.Path(__file__).parent / "snapshots" / "normalizer_cli_help.txt"


def _normalise(text: str) -> str:
    # Python < 3.10 titles the section "optional arguments:".
    return text.replace("optional arguments:", "options:").rstrip() + "\n"


@pytest.fixture(autouse=True)
def fixed_width(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")


def test_help_matches_snapshot():
    help_text = normalizer.build_parser().format_help()
    assert _normalise(help_text) == _normalise(SNAPSHOT.read_text())


def test_help_flag_exits_zero_and_prints_snapshot(capsys):
    with pytest.raises(SystemExit) as exc:
        normalizer.main(["--help"])
    assert exc.value.code == 0
    assert _normalise(capsys.readouterr().out) == _normalise(SNAPSHOT.read_text())


def test_help_documents_every_option():
    help_text = normalizer.build_parser().format_help()
    for token in ("--hops", "input", "-h, --help"):
        assert token in help_text


def test_unknown_flag_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        normalizer.main(["--nope"])
    assert exc.value.code == 2
    assert "usage: python -m astroml.ingestion.normalizer" in capsys.readouterr().err
