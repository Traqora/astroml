"""Coverage tests for docs/TROUBLESHOOTING.md.

The troubleshooting guide is the first place a user lands when an install or
configuration step fails on a clean machine. These tests pin the topics it
must cover so the v1 onboarding path stays self-contained: the known-broken
convenience targets, the environment/config pitfalls called out in
ONBOARDING.md, and pointers to the sibling docs.

The "expected path" is that the shipped guide covers every required topic; the
"relevant failure case" is that a guide missing one is reported, not silently
merged.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "TROUBLESHOOTING.md"

#: Topics the troubleshooting guide must cover. Each entry is matched as a
#: case-insensitive substring so the test survives minor rewording but still
#: pins the specific failure each row is about.
REQUIRED_TOPICS: dict[str, str] = {
    # Install failures -- the known-broken convenience targets and the fix.
    "make install is broken": "make install",
    "make dev-setup is broken": "make dev-setup",
    "make test collection aborts": "make test",
    "working editable dev install": "pip install -e",
    "module import errors": "modulenotfounderror",
    "pip resolution-too-deep": "resolution-too-deep",
    "optional api deps": "requirements-api.txt",
    # Configuration failures -- the pitfalls from ONBOARDING.md.
    "ASTROML_DATABASE_URL variable name": "astroml_database_url",
    "DATABASE_URL ignored": "database_url",
    "empty default password": "empty",
    "compose network host": "postgres_host",
    "recreate stale volume": "down -v",
    "database.yaml config": "config/database.yaml",
}

#: Headings and cross-links that make the guide discoverable as part of v1.
REQUIRED_STRUCTURE: dict[str, str] = {
    "supported boundaries section": "## Supported boundaries",
    "install failures section": "## Installation failures",
    "configuration failures section": "## Configuration failures",
    "link to onboarding": "onboarding.md",
    "link to postgres requirements": "ci_database_requirements.md",
    "link to roadmap": "roadmap.md",
}


def coverage_gaps(doc_text: str, required: dict[str, str]) -> list[str]:
    """Return the labels of required topics not present (case-insensitive)."""
    lowered = doc_text.lower()
    return [label for label, token in required.items() if token.lower() not in lowered]


def test_troubleshooting_covers_all_required_topics() -> None:
    # Happy path: the shipped guide covers every install and configuration
    # failure called out in ONBOARDING.md.
    text = DOC.read_text(encoding="utf-8")
    assert coverage_gaps(text, REQUIRED_TOPICS) == []
    assert coverage_gaps(text, REQUIRED_STRUCTURE) == []


def test_missing_topic_is_reported() -> None:
    # Failure path: a guide that drops the ASTROML_DATABASE_URL pitfall is
    # flagged, so the wrong variable name is never left undocumented.
    text = DOC.read_text(encoding="utf-8").replace("ASTROML_DATABASE_URL", "XXX")
    gaps = coverage_gaps(text, REQUIRED_TOPICS)
    assert "ASTROML_DATABASE_URL variable name" in gaps


def test_missing_section_is_reported() -> None:
    # Failure path: removing the Supported boundaries heading is flagged.
    text = DOC.read_text(encoding="utf-8").replace("## Supported boundaries", "## Notes")
    assert "supported boundaries section" in coverage_gaps(text, REQUIRED_STRUCTURE)
