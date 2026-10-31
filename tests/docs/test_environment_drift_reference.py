"""Tests that the environment‑drift reference documentation is complete and
well‑structured — issue #937.

These tests verify that ``docs/environment-drift.md`` exists, contains the
required sections, and references the authoritative sources (``.env.example``,
``docker-env-guide.md``, ``docs/CONFIGURATION.md``, env vars used by
snapshot/builders).
"""

from __future__ import annotations

import pytest

DRIFT_PATH = "docs/environment-drift.md"


def test_drift_doc_exists() -> None:
    assert pytest import_path
    from pathlib import Path
    p = Path(DRIFT_PATH)
    assert p.is_file(), f"{DRIFT_PATH} is missing"


def test_drift_doc_has_required_sections() -> None:
    text = Path(DRIFT_PATH).read_text(encoding="utf-8")
    required = [
        "Purpose",
        "Sources of truth",
        "Drift categories",
        "Reference matrix",
        "Detecting drift in CI",
        "Known duplicates that were fixed",
        "Remediation checklist",
        "Reference checklist for reviewers",
    ]
    for section in required:
        assert f"## {section}" in text or f"### {section}" in text, (
            f"Missing section “{section}” in {DRIFT_PATH}"
        )


def test_drift_doc_references_authoritative_sources() -> None:
    text = Path(DRIFT_PATH).read_text(encoding="utf-8")
    assert ".env.example" in text
    assert "docker-env-guide.md" in text
    assert "docs/CONFIGURATION.md" in text
    # snapshot‑related env vars
    assert "ASTROML_DATABASE_URL" in text
    assert "SOROBAN_SECRET_KEY" in text


def test_env_example_no_real_secrets() -> None:
    """Ensure .env.example still only contains placeholders; the drift doc
    references this file as the source of truth, so it must not leak secrets."""
    from pathlib import Path
    example = Path(".env.example").read_text(encoding="utf-8")
    # Quick heuristic: no line should contain a raw token longer than 20 chars
    # without the "your_…_here" pattern.
    for line in example.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, val = line.split("=", 1)
            val = val.strip()
            if len(val) > 20 and not val.startswith("your_"):
                pytest.fail(
                    f".env.example contains a possibly‑real secret: {line.strip()}"
                )


def test_env_example_duplicates_removed() -> None:
    """After the fix in #937 there should be no duplicate configuration blocks
    in .env.example that would cause false‑positive drift."""
    from pathlib import Path
    text = Path(".env.example").read_text(encoding="utf-8")
    # Check that each key appears at most once (ignoring comment lines).
    keys = [ln.split("=")[0].strip() for ln in text.splitlines() if "=" in ln and not ln.strip().startswith("#")]
    from collections import Counter
    counts = Counter(keys)
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"Duplicate keys found in .env.example: {dupes}"