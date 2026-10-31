"""Regression test for the disaster-recovery drill runbook (issue #890).

A DR runbook is part of the recovery plan, so it is worth guarding the same
way as code: the drill document must exist, cover the agreed sections, be
linked from the docs navigation and the top-level disaster-recovery page,
and every relative link in it must resolve. A stale or dangling runbook is
discovered here rather than during an incident.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
DRILL_PATH = DOCS_DIR / "runbooks" / "disaster_recovery_drill.md"
POSTGRES_RUNBOOK = DOCS_DIR / "runbooks" / "postgres_disaster_recovery.md"
DISASTER_RECOVERY_DOC = DOCS_DIR / "disaster-recovery.md"
MKDOCS = DOCS_DIR / "mkdocs.yml"

_REQUIRED_SECTIONS = [
    "# Disaster-Recovery Drill Runbook",
    "## Scope",
    "## 1. Cadence and roles",
    "## 3. Drill procedure",
    "## 5. Abort criteria",
    "## 6. Post-drill",
    "## 7. Drill report template",
]

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _relative_markdown_links(text: str) -> list[str]:
    links = []
    for raw in _LINK_RE.findall(text):
        target = raw.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith(("mailto:", "/", "#")):
            continue
        links.append(target)
    return links


def test_drill_runbook_exists():
    assert DRILL_PATH.is_file(), f"missing drill runbook at {DRILL_PATH}"


def test_drill_runbook_has_expected_sections():
    text = DRILL_PATH.read_text()
    missing = [section for section in _REQUIRED_SECTIONS if section not in text]
    assert not missing, f"drill runbook is missing sections: {missing}"


def test_drill_runbook_links_resolve():
    text = DRILL_PATH.read_text()
    base = DRILL_PATH.parent
    dangling = []
    for link in _relative_markdown_links(text):
        if not (base / link).resolve().is_file():
            dangling.append(link)
    assert not dangling, f"drill runbook has dangling links: {dangling}"


def test_drill_runbook_references_the_postgres_runbook():
    assert "postgres_disaster_recovery.md" in DRILL_PATH.read_text()


@pytest.mark.parametrize(
    "doc_path",
    [POSTGRES_RUNBOOK, DISASTER_RECOVERY_DOC],
)
def test_related_docs_exist(doc_path: Path):
    assert doc_path.is_file(), f"related doc missing: {doc_path}"


def test_drill_runbook_is_in_the_nav():
    nav = MKDOCS.read_text()
    assert "runbooks/disaster_recovery_drill.md" in nav


def test_disaster_recovery_doc_links_to_the_drill():
    text = DISASTER_RECOVERY_DOC.read_text()
    assert "runbooks/disaster_recovery_drill.md" in text


def test_postgres_runbook_link_still_resolves():
    text = POSTGRES_RUNBOOK.read_text()
    base = POSTGRES_RUNBOOK.parent
    for link in _relative_markdown_links(text):
        assert (base / link).resolve().is_file(), f"postgres runbook dangling link: {link}"
