"""Documentation contract for reproducing a saved v1 run (issue #1223)."""

from __future__ import annotations

import pytest

from tests.docs._helpers import assert_sections, doc_text, repo_path

DOC = "docs/run-artifacts.md"

REQUIRED_SECTIONS = [
    "Reproducing a saved run",
    "Steps",
    "Failure cases",
    "Boundaries",
]

REQUIRED_TOKENS = [
    "read_run_artifacts",
    "write_run_artifacts",
    "compute_fingerprint",
    'saved["run"]["config"]',
    'saved["run"]["inputs"]',
    'saved["run"]["fingerprint"]',
    "RunArtifactError",
    "InvalidRunInputError",
    "RunConflictError",
    "same_inputs is True",
    "local filesystem",
    "without torch, MLflow",
    "Bit-for-bit reproducibility is not a v1 guarantee",
]

REPRODUCTION_HEADING = "## Reproducing a saved run (issue #1223)\n"


def _doc_text() -> str:
    assert repo_path(DOC).is_file(), f"{DOC} is missing"
    return doc_text(DOC)


def test_reproduction_section_has_the_required_structure() -> None:
    text = _doc_text()
    assert_sections(text, REQUIRED_SECTIONS, DOC)


def test_reproduction_section_documents_the_supported_contract() -> None:
    text = _doc_text()
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    assert not missing, f"{DOC} does not document: {missing}"


def test_reproduction_section_is_linked_from_navigation() -> None:
    assert "run-artifacts.md" in doc_text("docs/index.md")
    assert "run-artifacts.md" in doc_text("docs/mkdocs.yml")


def test_missing_reproduction_section_is_detected() -> None:
    text = _doc_text()
    stale = text.replace(REPRODUCTION_HEADING, "")
    assert stale != text, "test fixture no longer matches the document heading"

    with pytest.raises(AssertionError, match="Reproducing a saved run"):
        assert_sections(stale, REQUIRED_SECTIONS, DOC)
