"""Multi-modal API endpoints (issue #631).

Provides REST endpoints for multi-modal encoding, fusion, inference,
and cross-modal retrieval.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from astroml.training.multimodal import (
    CrossModalRetriever,
    FusionMethod,
    Modality,
    MultiModalConfig,
    MultiModalDataBatch,
    MultiModalPipeline,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory pipeline (replace with persistent store in production)
# ---------------------------------------------------------------------------

_pipeline: MultiModalPipeline | None = None


def _ensure_pipeline() -> MultiModalPipeline:
    """Return (and lazily create) the global multi-modal pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MultiModalPipeline(
            MultiModalConfig(
                enabled_modalities=[Modality.TEXT, Modality.IMAGE, Modality.TABULAR],
                fusion_method=FusionMethod.CONCAT,
                cross_modal_retrieval=True,
            )
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ConfigRequest(BaseModel):
    """Request body for updating pipeline configuration."""

    enabled_modalities: list[str] | None = Field(
        default=None,
        description="List of modalities: text, image, tabular",
    )
    fusion_method: str | None = Field(
        default=None,
        description="Fusion method: concat, attention, gated, sum, mean, max",
    )
    fusion_output_dim: int | None = Field(default=None, ge=32, le=4096)


class InferenceRequest(BaseModel):
    """Request body for multi-modal inference."""

    text_inputs: list[str] | None = Field(
        default=None,
        description="List of text strings",
    )
    image_inputs: list[list[list[list[float]]]] | None = Field(
        default=None,
        description="List of image arrays (B, C, H, W) as nested lists",
    )
    tabular_inputs: list[list[float]] | None = Field(
        default=None,
        description="List of tabular feature vectors (B, F)",
    )


class EmbeddingResponse(BaseModel):
    """Response containing fused embeddings."""

    embeddings: list[list[float]]
    shape: list[int]
    fusion_method: str
    modalities_used: list[str]


class CrossModalRequest(BaseModel):
    """Request body for cross-modal retrieval."""

    query_text: str | None = None
    query_image: list[list[list[float]]] | None = None
    query_tabular: list[float] | None = None
    target_modality: str = "text"
    top_k: int = Field(default=5, ge=1, le=100)


class CrossModalResponse(BaseModel):
    """Response containing cross-modal retrieval results."""

    target_modality: str
    top_k: int
    indices: list[int]
    similarities: list[float]


class TrainingRequest(BaseModel):
    """Request body for multi-modal training."""

    epochs: int = Field(default=10, ge=1, le=1000)
    samples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Training samples with optional text, image, tabular, label fields",
    )


class TrainingResponse(BaseModel):
    """Response after multi-modal training."""

    epochs_completed: int
    final_loss: float | None
    modalities: list[str]


class PipelineInfoResponse(BaseModel):
    """Response describing the current pipeline state."""

    enabled_modalities: list[str]
    fusion_method: str
    fusion_output_dim: int
    embedding_dims: dict[str, int]
    cross_modal_enabled: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/multimodal/info",
    response_model=PipelineInfoResponse,
    tags=["multimodal"],
)
def get_pipeline_info() -> PipelineInfoResponse:
    """Return information about the current multi-modal pipeline."""
    pipe = _ensure_pipeline()
    dims = pipe.get_embedding_dims()
    return PipelineInfoResponse(
        enabled_modalities=[m.value for m in pipe.config.enabled_modalities],
        fusion_method=pipe.config.fusion_method.value,
        fusion_output_dim=pipe.config.fusion_output_dim,
        embedding_dims={k.value: v for k, v in dims.items()},
        cross_modal_enabled=pipe.config.cross_modal_retrieval,
    )


@router.post(
    "/api/v1/multimodal/config",
    response_model=PipelineInfoResponse,
    tags=["multimodal"],
)
def update_config(req: ConfigRequest) -> PipelineInfoResponse:
    """Update the multi-modal pipeline configuration."""
    global _pipeline
    config_dict = _pipeline.config.model_dump() if _pipeline else {}

    if req.enabled_modalities is not None:
        config_dict["enabled_modalities"] = [Modality(m) for m in req.enabled_modalities]
    if req.fusion_method is not None:
        config_dict["fusion_method"] = FusionMethod(req.fusion_method)
    if req.fusion_output_dim is not None:
        config_dict["fusion_output_dim"] = req.fusion_output_dim

    new_config = MultiModalConfig(**config_dict)
    _pipeline = MultiModalPipeline(new_config)

    return get_pipeline_info()


@router.post(
    "/api/v1/multimodal/encode",
    response_model=EmbeddingResponse,
    tags=["multimodal"],
)
def encode(req: InferenceRequest) -> EmbeddingResponse:
    """Encode multi-modal inputs and return fused embeddings."""
    pipe = _ensure_pipeline()

    import numpy as np

    batch = MultiModalDataBatch(
        text_inputs=req.text_inputs,
        image_inputs=np.array(req.image_inputs, dtype=np.float32) if req.image_inputs else None,
        tabular_inputs=np.array(req.tabular_inputs, dtype=np.float32) if req.tabular_inputs else None,
    )

    embeddings = pipe.infer(batch)
    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        shape=list(embeddings.shape),
        fusion_method=pipe.config.fusion_method.value,
        modalities_used=[m.value for m in pipe.config.enabled_modalities],
    )


@router.post(
    "/api/v1/multimodal/retrieve",
    response_model=CrossModalResponse,
    tags=["multimodal"],
)
def cross_modal_retrieve(req: CrossModalRequest) -> CrossModalResponse:
    """Retrieve items from a target modality given a query in another modality."""
    pipe = _ensure_pipeline()
    if not pipe.config.cross_modal_retrieval:
        raise HTTPException(status_code=400, detail="Cross-modal retrieval not enabled")

    import numpy as np

    # Encode the query
    batch = MultiModalDataBatch(
        text_inputs=[req.query_text] if req.query_text else None,
        image_inputs=np.array([req.query_image], dtype=np.float32) if req.query_image else None,
        tabular_inputs=np.array([req.tabular_inputs], dtype=np.float32).reshape(1, -1)
        if req.tabular_inputs
        else None,
    )

    query_embedding = pipe.infer(batch)[0]
    target = Modality(req.target_modality)

    indices, similarities = pipe.retrieve_cross_modal(query_embedding, target, req.top_k)
    return CrossModalResponse(
        target_modality=target.value,
        top_k=req.top_k,
        indices=indices.tolist(),
        similarities=similarities.tolist(),
    )


@router.post(
    "/api/v1/multimodal/train",
    response_model=TrainingResponse,
    tags=["multimodal"],
)
def train_multimodal(req: TrainingRequest) -> TrainingResponse:
    """Train the multi-modal pipeline on provided samples."""
    pipe = _ensure_pipeline()
    import numpy as np

    batches: list[MultiModalDataBatch] = []
    for sample in req.samples:
        batch = MultiModalDataBatch(
            text_inputs=sample.get("text") if isinstance(sample.get("text"), list) else [sample.get("text")] if sample.get("text") else None,
            image_inputs=np.array(sample.get("image"), dtype=np.float32) if sample.get("image") is not None else None,
            tabular_inputs=np.array(sample.get("tabular"), dtype=np.float32) if sample.get("tabular") is not None else None,
            labels=np.array([sample.get("label")], dtype=np.int32) if sample.get("label") is not None else None,
        )
        batches.append(batch)

    result = pipe.train(batches, epochs=req.epochs)
    return TrainingResponse(
        epochs_completed=result["epochs_completed"],
        final_loss=result["final_loss"],
        modalities=result["modalities"],
    )