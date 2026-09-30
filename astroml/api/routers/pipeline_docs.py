"""Pipeline documentation API: generate, version and publish pipeline docs.

Resolves part of #646.  Mounted at ``/api/v1/pipeline-docs``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from astroml.pipeline.documentation.data_flow import NodeKind
from astroml.pipeline.documentation.generator import (
    IOSpec,
    PipelineDocGenerator,
    PipelineMetadata,
    StageMetadata,
)
from astroml.pipeline.documentation.model_card import ModelCardBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pipeline-docs", tags=["pipeline-docs"])

_DOC_ROOT = Path("docs/pipelines")
_generator = PipelineDocGenerator(_DOC_ROOT)


def get_generator() -> PipelineDocGenerator:
    """Return the generator backing this router (FastAPI dependency / test hook)."""
    return _generator


# ─── Schemas ─────────────────────────────────────────────────────────────────


class StageRequest(BaseModel):
    """One pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: NodeKind = NodeKind.TRANSFORM
    description: str = ""
    implementation: str = ""
    depends_on: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class IOSpecRequest(BaseModel):
    """A pipeline input or output."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: str = "dataset"
    location: str = ""
    schema_fields: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class ModelCardRequest(BaseModel):
    """Minimal model card payload attached to a pipeline."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    overview: str = ""
    version: str = "0.1.0"
    owners: list[str] = Field(default_factory=list)
    primary_uses: list[str] = Field(default_factory=list)
    out_of_scope_uses: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    ethical_considerations: list[str] = Field(default_factory=list)


class PipelineDocRequest(BaseModel):
    """Request to generate documentation for a pipeline."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    version: str = "0.1.0"
    owner: str = ""
    framework: str = ""
    schedule: str = "on-demand"
    stages: list[StageRequest] = Field(default_factory=list)
    inputs: list[IOSpecRequest] = Field(default_factory=list)
    outputs: list[IOSpecRequest] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    model_card: ModelCardRequest | None = None
    publish: bool = False


def _to_metadata(request: PipelineDocRequest) -> PipelineMetadata:
    """Convert an API request into :class:`PipelineMetadata`."""
    card = None
    if request.model_card is not None:
        builder = (
            ModelCardBuilder(request.model_card.name, version=request.model_card.version)
            .with_overview(request.model_card.overview)
            .with_owners(request.model_card.owners)
            .with_intended_use(
                primary_uses=request.model_card.primary_uses,
                out_of_scope_uses=request.model_card.out_of_scope_uses,
            )
            .with_metrics(request.model_card.metrics)
            .with_ethical_considerations(request.model_card.ethical_considerations)
        )
        card = builder.build()

    return PipelineMetadata(
        name=request.name,
        description=request.description,
        version=request.version,
        owner=request.owner,
        framework=request.framework,
        schedule=request.schedule,
        stages=[
            StageMetadata(
                name=stage.name,
                kind=stage.kind,
                description=stage.description,
                implementation=stage.implementation,
                depends_on=list(stage.depends_on),
                parameters=dict(stage.parameters),
            )
            for stage in request.stages
        ],
        inputs=[
            IOSpec(
                name=spec.name,
                kind=spec.kind,
                location=spec.location,
                schema=dict(spec.schema_fields),
                description=spec.description,
            )
            for spec in request.inputs
        ],
        outputs=[
            IOSpec(
                name=spec.name,
                kind=spec.kind,
                location=spec.location,
                schema=dict(spec.schema_fields),
                description=spec.description,
            )
            for spec in request.outputs
        ],
        parameters=dict(request.parameters),
        dependencies=list(request.dependencies),
        model_card=card,
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/generate", status_code=201)
async def generate_docs(request: PipelineDocRequest) -> dict[str, Any]:
    """Generate (and optionally publish) documentation for a pipeline."""
    generator = get_generator()
    try:
        doc = generator.generate(_to_metadata(request))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload: dict[str, Any] = {
        "pipeline": doc.metadata.name,
        "version": doc.version.to_dict(),
        "is_complete": doc.is_complete,
        "validation_problems": list(doc.validation_problems),
        "markdown": doc.markdown,
        "mermaid": doc.mermaid,
    }
    if request.publish:
        payload["published_to"] = str(generator.publish(doc))
    return payload


@router.post("/diagram")
async def generate_diagram(
    request: PipelineDocRequest,
    fmt: Literal["mermaid", "dot", "json"] = "mermaid",
) -> dict[str, Any]:
    """Render only the data flow diagram for a pipeline."""
    try:
        diagram = _to_metadata(request).to_diagram()
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    renderers = {
        "mermaid": diagram.to_mermaid,
        "dot": diagram.to_dot,
        "json": diagram.to_dict,
    }
    return {
        "format": fmt,
        "diagram": renderers[fmt](),
        "problems": diagram.validate(),
    }


@router.post("/model-card", status_code=201)
async def generate_model_card(request: ModelCardRequest) -> dict[str, Any]:
    """Generate a Google-spec model card in JSON and Markdown."""
    card = (
        ModelCardBuilder(request.name, version=request.version)
        .with_overview(request.overview)
        .with_owners(request.owners)
        .with_intended_use(
            primary_uses=request.primary_uses,
            out_of_scope_uses=request.out_of_scope_uses,
        )
        .with_metrics(request.metrics)
        .with_ethical_considerations(request.ethical_considerations)
        .build()
    )
    return {
        "card": card.to_dict(),
        "markdown": card.to_markdown(),
        "is_valid": card.is_valid(),
        "problems": card.validate(),
    }


@router.get("/versions/{pipeline_name}")
async def pipeline_versions(pipeline_name: str) -> dict[str, Any]:
    """Return the published documentation version history of a pipeline."""
    versions = get_generator().versions(pipeline_name)
    if not versions:
        raise HTTPException(
            status_code=404, detail=f"no published documentation for {pipeline_name!r}"
        )
    return {
        "pipeline": pipeline_name,
        "version_count": len(versions),
        "versions": [version.to_dict() for version in versions],
    }


@router.post("/index", status_code=201)
async def rebuild_index() -> dict[str, str]:
    """Rebuild the documentation site index page."""
    return {"index": str(get_generator().build_index())}
