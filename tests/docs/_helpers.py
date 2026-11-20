"""Shared helpers for documentation assertion tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_path(relative: str) -> Path:
    """Return an absolute path inside the repository root."""
    return ROOT / relative


def doc_text(relative: str) -> str:
    """Read a tracked documentation or configuration file as text."""
    return repo_path(relative).read_text(encoding="utf-8")


def assert_sections(text: str, sections: list[str], source: str) -> None:
    """Assert that every named section heading is present in ``text``."""
    for section in sections:
        assert (
            f"## {section}" in text or f"### {section}" in text
        ), f"Missing section '{section}' in {source}"
