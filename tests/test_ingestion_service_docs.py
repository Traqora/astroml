"""Docs-generation regression tests for the ingestion pipeline (issue #981).

Ensures every public API on ``astroml.ingestion.service`` carries a
docstring that documents its parameters and return/yield value, so
generated API docs stay complete.
"""

import inspect

import pytest

from astroml.ingestion import service
from astroml.ingestion.service import IngestionService

PUBLIC_CLASSES = [service.IngestionResult, service.LedgerOutcome, IngestionService]
PUBLIC_METHODS = [
    name
    for name, member in inspect.getmembers(IngestionService, inspect.isfunction)
    if not name.startswith("_") and member.__qualname__.startswith("IngestionService.")
]


def test_module_has_docstring():
    assert inspect.getdoc(service)


@pytest.mark.parametrize("cls", PUBLIC_CLASSES, ids=lambda c: c.__name__)
def test_public_classes_documented(cls):
    assert inspect.getdoc(cls)


def test_expected_pipeline_methods_present():
    assert {"ingest", "ingest_stream", "ingest_incremental", "ingest_backfill_chunked",
            "get_status"} <= set(PUBLIC_METHODS)


@pytest.mark.parametrize("name", PUBLIC_METHODS)
def test_public_methods_document_args_and_output(name):
    method = getattr(IngestionService, name)
    doc = inspect.getdoc(method)
    assert doc, f"{name} missing docstring"

    params = [p for p in inspect.signature(inspect.unwrap(method)).parameters if p != "self"]
    if params:
        assert "Args:" in doc, f"{name} docstring missing Args section"
        for param in params:
            assert param in doc, f"{name} docstring does not document {param!r}"

    assert "Returns" in doc or "Yields" in doc, f"{name} missing Returns/Yields"
