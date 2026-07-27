"""Explanation router — LLM-powered explanation endpoints.

Resolves #457: Generate human-readable explanations for model predictions,
transactions, and fraud decisions using LLM.
"""
from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.services.llm import LLMService
from api.routers.llm import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm/explanations", tags=["llm", "explanations"])


class ExplainRequest(BaseModel):
    subject: str = Field(..., description="What to explain: 'fraud_decision', 'model_prediction', 'transaction'")
    context: dict[str, Any] = Field(..., description="Contextual data for the explanation")
    audience: str = Field("user", pattern="^(user|analyst|regulator)$", description="Target audience")
    model: str = Field("gpt-4-turbo")
    max_words: int = Field(150, ge=20, le=500)


class ExplainResponse(BaseModel):
    id: str
    subject: str
    explanation: str
    audience: str
    latency_ms: float
    cost: float


@router.post(
    "/",
    response_model=ExplainResponse,
    summary="Generate an LLM explanation",
    operation_id="llm_explain",
)
async def explain(
    body: ExplainRequest,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> ExplainResponse:
    """Generate a human-readable explanation for a model decision or transaction."""
    audience_note = {
        "user": "Write for a non-technical end user. Use plain language.",
        "analyst": "Write for a data analyst. Include technical details.",
        "regulator": "Write for a regulatory compliance officer. Be precise and cite data.",
    }.get(body.audience, "")

    prompt = (
        f"Explain the following {body.subject} in under {body.max_words} words.\n"
        f"{audience_note}\n\n"
        f"Context:\n{body.context}\n\n"
        f"Explanation:"
    )
    user_id = getattr(request.state, "user_id", None)
    try:
        result = await service.generate(prompt=prompt, model=body.model, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid request") from exc

    return ExplainResponse(
        id=result["id"],
        subject=body.subject,
        explanation=result["content"],
        audience=body.audience,
        latency_ms=result["latency_ms"],
        cost=result["cost"],
    )


@router.post(
    "/fraud",
    response_model=ExplainResponse,
    summary="Explain a fraud detection decision",
    operation_id="llm_explain_fraud",
)
async def explain_fraud(
    body: ExplainRequest,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> ExplainResponse:
    """Specialised endpoint for fraud decision explanations."""
    body.subject = "fraud_decision"
    return await explain(body, request, service)
