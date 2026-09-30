"""Documentation-completeness regression test for issue #969.

``astroml/ingestion/service.py`` previously documented what the ingestion
service *is* (key components, dependencies) but not the operational fields
a "model card"-style standard expects: intended use, limitations/guarantees,
and pointers to the tests that cover the described behavior. This parses
the checked-in module docstring and asserts those sections are present and
non-empty, so a future edit can't silently drop or blank them again.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = ROOT / "astroml" / "ingestion" / "service.py"


def _module_docstring() -> str:
    tree = ast.parse(SERVICE_PATH.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree)
    assert doc, "astroml/ingestion/service.py must have a module docstring"
    return doc


def test_module_docstring_documents_intended_use():
    doc = _module_docstring()
    assert "Intended use" in doc

    section = doc.split("Intended use", 1)[1].split("Limitations", 1)[0]
    assert section.strip(), "Intended use section must not be empty"


def test_module_docstring_documents_limitations():
    doc = _module_docstring()
    assert "Limitations" in doc

    section = doc.split("Limitations", 1)[1].split("Test coverage", 1)[0]
    assert section.strip(), "Limitations section must not be empty"
    # The most operationally relevant gap for callers of this service.
    assert "retry" in section.lower()


def test_module_docstring_points_to_test_coverage():
    doc = _module_docstring()
    assert "Test coverage" in doc

    section = doc.split("Test coverage", 1)[1]
    assert "tests/" in section
