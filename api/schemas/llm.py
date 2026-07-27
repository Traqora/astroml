"""Pydantic schemas for LLM API — requests, responses, and error formats.

Resolves #457: All request/response models for the LLM API gateway
with idempotency keys, pagination, and consistent error shapes.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
import uuid


# ─── Shared ─────────────────────────────────────────────────────────────────

class UsageInfo(BaseModel):
    prompt_tokens: int = Field(0, description="Tokens in the prompt")
    completion_tokens: int = Field(0, description="Tokens in the completion")
    total_tokens: int = Field(0, description="Total tokens consumed")


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["RATE_LIMIT_EXCEEDED"])
    message: str = Field(..., examples=["Rate limit exceeded. Retry after 60s."])
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ─── Generation ─────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request body for POST /api/v1/llm/generate."""

    prompt: str = Field(..., min_length=1, max_length=32_000, description="Input prompt")
    model: str = Field("gpt-4-turbo", description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for safe retries",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response body for POST /api/v1/llm/generate."""

    id: str = Field(default_factory=lambda: f"gen_{uuid.uuid4().hex[:8]}")
    model: str
    content: str
    usage: UsageInfo
    cost: float = Field(0.0, description="Estimated cost in USD")
    latency_ms: float = Field(0.0, description="End-to-end latency in milliseconds")
    cached: bool = False


# ─── Embeddings ─────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    """Request body for POST /api/v1/llm/embed."""

    input: str | list[str] = Field(..., description="Text or list of texts to embed")
    model: str = Field("text-embedding-3-small", description="Embedding model")


class EmbedResponse(BaseModel):
    """Response body for POST /api/v1/llm/embed."""

    id: str = Field(default_factory=lambda: f"emb_{uuid.uuid4().hex[:8]}")
    model: str
    embeddings: list[list[float]]
    usage: UsageInfo
    cost: float = 0.0


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /api/v1/llm/chat."""

    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str = Field("gpt-4-turbo")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    stream: bool = Field(False, description="If true, returns SSE stream")
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response body for POST /api/v1/llm/chat."""

    id: str = Field(default_factory=lambda: f"chat_{uuid.uuid4().hex[:8]}")
    model: str
    message: ChatMessage
    usage: UsageInfo
    cost: float = 0.0
    latency_ms: float = 0.0


# ─── RAG Query ───────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    """Request body for POST /api/v1/llm/rag/query."""

    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    model: str = Field("gpt-4-turbo")
    collection: str | None = Field(None, description="Knowledge base collection name")
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


class RAGDocument(BaseModel):
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Response body for POST /api/v1/llm/rag/query."""

    id: str = Field(default_factory=lambda: f"rag_{uuid.uuid4().hex[:8]}")
    query: str
    answer: str
    documents: list[RAGDocument]
    usage: UsageInfo
    cost: float = 0.0
    latency_ms: float = 0.0


# ─── Models list ─────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    provider: str
    context_window: int
    cost_per_1k_prompt_tokens: float
    cost_per_1k_completion_tokens: float
    supports_streaming: bool = True
    supports_vision: bool = False


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]
    total: int


# ─── Cost / Usage ─────────────────────────────────────────────────────────────

class CostUsageResponse(BaseModel):
    """Response body for GET /api/v1/llm/cost/usage."""

    user_id: str
    period: str = Field(..., description="e.g. '2026-07' for monthly")
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    cost_by_model: dict[str, float]
    cost_by_day: list[dict[str, Any]] = Field(default_factory=list)


# ─── WebSocket streaming ──────────────────────────────────────────────────────

class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""

    id: str
    delta: str
    finish_reason: str | None = None
    usage: UsageInfo | None = None
