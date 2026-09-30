from astroml.utils.exceptions import AstroMLError
"""Data contract API endpoints for AstroML.

Provides endpoints to validate data against contracts, infer contracts from data,
query breach history, and verify pipeline stages.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from astroml.pipeline.contracts.quality_contract import QualityContract
from astroml.pipeline.contracts.schema_contract import SchemaContract
from astroml.pipeline.contracts.semantic_contract import SemanticContract
from astroml.pipeline.contracts.verifier import ContractVerifier, MissingContractError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/contracts", tags=["data-contracts"])

_verifier = ContractVerifier()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ColumnSchema(BaseModel):
    """Schema specification for a single column."""

    dtype: str | None = None
    nullable: bool = True
    unique: bool = False
    regex: str | None = None

    model_config = ConfigDict(extra="forbid")


class SchemaDefinition(BaseModel):
    """Schema definition for contract creation."""

    columns: dict[str, ColumnSchema]

    model_config = ConfigDict(extra="forbid")


class ValidateRequest(BaseModel):
    """Request body for validating data against a contract."""

    data: list[dict[str, Any]]
    schema_def: SchemaDefinition | None = None
    contract_type: str = Field(default="schema", pattern="^(schema|quality|semantic)$")

    model_config = ConfigDict(extra="forbid")


class ValidateResponse(BaseModel):
    """Response body for validation results."""

    is_valid: bool
    contract_type: str
    details: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class InferResponse(BaseModel):
    """Response body for inferred contract."""

    contract_type: str
    columns: dict[str, dict[str, Any]]
    row_count: int

    model_config = ConfigDict(extra="forbid")


class BreachItem(BaseModel):
    """Single breach record."""

    contract_name: str
    contract_type: str
    timestamp: str
    details: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class BreachListResponse(BaseModel):
    """Response body for breach history."""

    breaches: list[BreachItem]
    total: int

    model_config = ConfigDict(extra="forbid")


class PipelineStageSpec(BaseModel):
    """Specification for a pipeline stage."""

    contracts: list[str]

    model_config = ConfigDict(extra="forbid")


class PipelineVerifyRequest(BaseModel):
    """Request body for pipeline verification."""

    data: list[dict[str, Any]]
    stages: dict[str, PipelineStageSpec]
    strict: bool = False
    """If True, a stage referencing an unregistered contract name returns
    HTTP 422 instead of silently verifying nothing for that name. A CI
    caller using this endpoint as a build gate should set this to True: a
    misspelled or stale contract name would otherwise make the gate report
    `passed: true` for a stage that validated zero contracts."""

    model_config = ConfigDict(extra="forbid")


class PipelineVerifyResponse(BaseModel):
    """Response body for pipeline verification."""

    passed: bool
    stages: dict[str, dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/validate", response_model=ValidateResponse)
async def validate_data(body: ValidateRequest) -> ValidateResponse:
    """Validate data against a contract.

    Accepts a list of records and a schema definition or references a
    pre-registered contract.
    """
    if not body.data:
        raise HTTPException(status_code=400, detail="data must be a non-empty list")

    try:
        df = pd.DataFrame(body.data)
    except AstroMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")

    if body.schema_def is not None:
        schema_dict = {name: col.model_dump() for name, col in body.schema_def.columns.items()}
        if body.contract_type == "schema":
            contract = SchemaContract.from_schema({"columns": schema_dict}, name="api-validate")
        elif body.contract_type == "quality":
            contract = QualityContract(name="api-validate")
            for col_name, col_spec in schema_dict.items():
                if col_spec.get("dtype") == "numeric":
                    if "min" in col_spec:
                        contract.add_constraint(col_name, "range", {"min": col_spec["min"]})
                    if "max" in col_spec:
                        contract.add_constraint(col_name, "range", {"max": col_spec["max"]})
                if not col_spec.get("nullable", True):
                    contract.add_constraint(col_name, "null_ratio", {"max": 0.0})
            _verifier.add_contract(contract, "api-validate")
        elif body.contract_type == "semantic":
            contract = SemanticContract(name="api-validate")
            _verifier.add_contract(contract, "api-validate")
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown contract_type: {body.contract_type}"
            )

        _verifier.add_contract(contract, "api-validate")

    result = _verifier.verify(df, contract_names=["api-validate"])
    if result.results:
        detail = result.results[0]
        return ValidateResponse(
            is_valid=detail.passed,
            contract_type=body.contract_type,
            details=_serialize_result(detail.details),
        )

    return ValidateResponse(
        is_valid=False, contract_type=body.contract_type, details={"error": "No contracts matched"}
    )


@router.post("/infer", response_model=InferResponse)
async def infer_contract(body: ValidateRequest) -> InferResponse:
    """Infer a schema contract from data.

    Accepts a list of records and returns the inferred schema.
    """
    if not body.data:
        raise HTTPException(status_code=400, detail="data must be a non-empty list")

    try:
        df = pd.DataFrame(body.data)
    except AstroMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")

    contract = SchemaContract.from_dataframe(df, name="inferred")
    columns = {}
    for col_name, col_spec in contract.columns.items():
        columns[col_name] = {
            "dtype": col_spec["dtype"],
            "nullable": col_spec["nullable"],
        }

    return InferResponse(
        contract_type="schema",
        columns=columns,
        row_count=len(df),
    )


@router.get("/breaches", response_model=BreachListResponse)
async def get_breaches() -> BreachListResponse:
    """Get breach history from the contract verifier."""
    breaches = [
        BreachItem(
            contract_name=b.contract_name,
            contract_type=b.contract_type,
            timestamp=b.timestamp,
            details=b.details,
        )
        for b in _verifier.breach_history
    ]
    return BreachListResponse(breaches=breaches, total=len(breaches))


@router.post("/pipeline/verify", response_model=PipelineVerifyResponse)
async def verify_pipeline(body: PipelineVerifyRequest) -> PipelineVerifyResponse:
    """Verify pipeline stage contracts."""
    if not body.data:
        raise HTTPException(status_code=400, detail="data must be a non-empty list")

    try:
        df = pd.DataFrame(body.data)
    except AstroMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")

    # Build pipeline_stages format for the verifier
    pipeline_stages: dict[str, list[str]] = {}
    for stage_name, stage_spec in body.stages.items():
        pipeline_stages[stage_name] = stage_spec.contracts

    try:
        result = _verifier.verify_pipeline(df, pipeline_stages, strict=body.strict)
    except MissingContractError as e:
        raise HTTPException(status_code=422, detail=str(e))

    stages_dict: dict[str, dict[str, Any]] = {}
    for stage in result.stages:
        stages_dict[stage.stage_name] = {
            "passed": stage.passed,
            "contracts": [
                {
                    "name": r.name,
                    "contract_type": r.contract_type,
                    "passed": r.passed,
                }
                for r in stage.results
            ],
        }

    return PipelineVerifyResponse(
        passed=result.passed,
        stages=stages_dict,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_result(result: Any) -> dict[str, Any]:
    """Convert validation result to a serializable dict."""
    if isinstance(result, str):
        return {"error": result}
    if hasattr(result, "__dataclass_fields__"):
        output: dict[str, Any] = {}
        for field_name in result.__dataclass_fields__:
            value = getattr(result, field_name)
            if isinstance(value, (list, tuple)):
                output[field_name] = [_safe_serialize(v) for v in value]
            else:
                output[field_name] = _safe_serialize(value)
        return output
    if isinstance(result, dict):
        return {k: _safe_serialize(v) for k, v in result.items()}
    return {"value": str(result)}


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    try:
        if hasattr(value, "__dataclass_fields__"):
            return _serialize_result(value)
    except AstroMLError:
        pass
    return str(value)
