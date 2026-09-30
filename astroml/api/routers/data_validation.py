"""Data validation API: expectation suites, validation runs and data docs.

Resolves part of #644.  Mounted at ``/api/v1/data-validation``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from astroml.validation.great_expectations.data_docs import DataDocsBuilder
from astroml.validation.great_expectations.suite_builder import (
    ExpectationSuite,
    SuiteBuilder,
    great_expectations_available,
)
from astroml.validation.great_expectations.validator import (
    DataValidator,
    ValidationStore,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data-validation", tags=["data-validation"])

_SUITE_DIR = Path(os.environ.get("ASTROML_SUITE_DIR", "configs/validation/suites"))
_RESULT_DIR = Path(os.environ.get("ASTROML_VALIDATION_RESULTS", "validation_results"))
_DOCS_DIR = Path(os.environ.get("ASTROML_DATA_DOCS", "docs/data_docs"))

_suites: dict[str, ExpectationSuite] = {}
_store = ValidationStore(_RESULT_DIR)
_docs = DataDocsBuilder(_DOCS_DIR)


def get_store() -> ValidationStore:
    """Return the validation result store backing this router."""
    return _store


def get_docs_builder() -> DataDocsBuilder:
    """Return the data docs builder backing this router."""
    return _docs


def _resolve_suite(name: str) -> ExpectationSuite:
    """Return an in-memory suite, falling back to the suite directory."""
    suite = _suites.get(name)
    if suite is not None:
        return suite
    path = _SUITE_DIR / f"{name}.json"
    if path.is_file():
        suite = ExpectationSuite.load(path)
        _suites[name] = suite
        return suite
    raise HTTPException(status_code=404, detail=f"unknown expectation suite {name!r}")


# ─── Schemas ─────────────────────────────────────────────────────────────────


class ProfileRequest(BaseModel):
    """Generate an expectation suite by profiling a dataset."""

    model_config = ConfigDict(extra="forbid")

    suite_name: str = Field(min_length=1)
    data: dict[str, list[Any]] = Field(min_length=1)
    tolerance: float = Field(default=0.1, ge=0.0, lt=10.0)
    null_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)
    max_categories: int = Field(default=20, ge=1, le=1000)
    persist: bool = False


class SuiteRequest(BaseModel):
    """Register a hand-authored expectation suite."""

    model_config = ConfigDict(extra="forbid")

    suite: dict[str, Any]
    persist: bool = False


class ValidateRequest(BaseModel):
    """Validate a dataset against a registered suite."""

    model_config = ConfigDict(extra="forbid")

    suite_name: str = Field(min_length=1)
    data: dict[str, list[Any]] = Field(min_length=1)
    dataset_name: str = "dataset"
    store_result: bool = True
    build_docs: bool = False


# ─── Suite management ────────────────────────────────────────────────────────


@router.get("/status")
async def validation_status() -> dict[str, Any]:
    """Report which validation backend is active and what is registered."""
    return {
        "great_expectations_installed": great_expectations_available(),
        "suite_count": len(_suites),
        "suites": sorted(_suites),
        "suite_dir": str(_SUITE_DIR),
        "result_dir": str(_RESULT_DIR),
        "data_docs_dir": str(_DOCS_DIR),
    }


@router.post("/suites/profile", status_code=201)
async def profile_dataset(request: ProfileRequest) -> dict[str, Any]:
    """Profile a dataset and generate an expectation suite automatically."""
    try:
        suite = SuiteBuilder.from_dataset(
            request.suite_name,
            request.data,
            tolerance=request.tolerance,
            null_tolerance=request.null_tolerance,
            max_categories=request.max_categories,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _suites[suite.name] = suite
    payload: dict[str, Any] = {"suite": suite.to_dict(), "expectation_count": len(suite)}
    if request.persist:
        payload["path"] = str(suite.save(_SUITE_DIR / f"{suite.name}.json"))
    return payload


@router.post("/suites", status_code=201)
async def register_suite(request: SuiteRequest) -> dict[str, Any]:
    """Register a hand-authored expectation suite."""
    try:
        suite = ExpectationSuite.from_dict(request.suite)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _suites[suite.name] = suite
    payload: dict[str, Any] = {"suite_name": suite.name, "expectation_count": len(suite)}
    if request.persist:
        payload["path"] = str(suite.save(_SUITE_DIR / f"{suite.name}.json"))
    return payload


@router.get("/suites")
async def list_suites() -> dict[str, Any]:
    """List the registered expectation suites."""
    return {
        "suites": [
            {"name": name, "expectation_count": len(suite), "columns": suite.columns()}
            for name, suite in sorted(_suites.items())
        ]
    }


@router.get("/suites/{suite_name}")
async def get_suite(suite_name: str) -> dict[str, Any]:
    """Return one expectation suite."""
    return _resolve_suite(suite_name).to_dict()


@router.delete("/suites/{suite_name}", status_code=204)
async def delete_suite(suite_name: str) -> None:
    """Deregister an expectation suite."""
    if _suites.pop(suite_name, None) is None:
        raise HTTPException(status_code=404, detail=f"unknown suite {suite_name!r}")


# ─── Validation ──────────────────────────────────────────────────────────────


@router.post("/validate")
async def validate_dataset(request: ValidateRequest) -> dict[str, Any]:
    """Validate a dataset against a registered expectation suite."""
    suite = _resolve_suite(request.suite_name)
    result = DataValidator(suite).validate(request.data, dataset_name=request.dataset_name)

    payload = result.to_dict()
    payload["summary"] = result.summary()
    if request.store_result:
        payload["stored_at"] = str(get_store().save(result))
    if request.build_docs:
        payload["data_docs"] = str(
            get_docs_builder().build(suites=[suite], results=[result], store=get_store())
        )
    return payload


@router.get("/results/{suite_name}")
async def validation_history(suite_name: str) -> dict[str, Any]:
    """Return the stored validation history of a suite."""
    history = get_store().history(suite_name)
    if not history:
        raise HTTPException(status_code=404, detail=f"no stored results for suite {suite_name!r}")
    return {
        "suite_name": suite_name,
        "run_count": len(history),
        "latest": history[-1],
        "history": history,
    }


@router.get("/results/{suite_name}/{run_id}")
async def validation_result(suite_name: str, run_id: str) -> dict[str, Any]:
    """Return one stored validation result document."""
    try:
        return get_store().load(suite_name, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dashboard")
async def validation_dashboard() -> dict[str, Any]:
    """Return a dashboard view of the latest run for every stored suite."""
    store = get_store()
    entries = []
    for suite_name in store.suites():
        latest = store.latest(suite_name)
        if latest is not None:
            entries.append({"suite_name": suite_name, **latest})
    failing = [entry for entry in entries if not entry["success"]]
    return {
        "suite_count": len(entries),
        "failing_count": len(failing),
        "all_passing": not failing,
        "suites": sorted(entries, key=lambda entry: entry["success_percent"]),
    }


@router.post("/data-docs", status_code=201)
async def build_data_docs() -> dict[str, str]:
    """Rebuild the static data docs site from stored results."""
    store = get_store()
    suites = [_resolve_suite(name) for name in _suites]
    return {"data_docs": str(get_docs_builder().build(suites=suites, store=store))}
