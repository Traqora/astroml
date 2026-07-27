"""LLM API Gateway router — unified REST endpoints for all LLM features.

Resolves #457: Production-ready LLM API with rate limiting, authentication,
streaming, and comprehensive OpenAPI documentation.

Endpoints:
  POST /api/v1/llm/generate          — Text completion
  POST /api/v1/llm/generate/stream   — Streaming completion (SSE)
  POST /api/v1/llm/embed             — Embeddings
  POST /api/v1/llm/chat              — Chat completion
  POST /api/v1/llm/rag/query         — RAG query
  GET  /api/v1/llm/models            — List available models
  GET  /api/v1/llm/cost/usage        — Cost usage report
  WS   /api/v1/llm/chat/ws           — Streaming chat over WebSocket
  WS   /api/v1/llm/stream            — Generic streaming over WebSocket
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from api.schemas.llm import (
    GenerateRequest,
    GenerateResponse,
    EmbedRequest,
    EmbedResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage as SchemaChatMessage,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGDocument,
    ModelsListResponse,
    ModelInfo,
    CostUsageResponse,
    UsageInfo,
    StreamChunk,
    ErrorResponse,
    ErrorDetail,
)
from api.services.llm import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])

# Shared service instance (dependency-injectable for testing)
_llm_service = LLMService()


def get_llm_service() -> LLMService:
    return _llm_service


def _get_user_id(request: Request) -> str | None:
    """Extract user ID from request state (set by AuthMiddleware)."""
    return getattr(request.state, "user_id", None)


# ─── REST: Generate ─────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    summary="Generate a text completion",
    operation_id="llm_generate",
)
async def generate_completion(
    body: GenerateRequest,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> GenerateResponse:
    """Generate an LLM completion from a prompt.

    Enforces safety guardrails, rate limits, and logs to the audit trail.
    """
    user_id = _get_user_id(request)
    try:
        result = await service.generate(
            prompt=body.prompt,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            user_id=user_id,
            idempotency_key=body.idempotency_key,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid request") from exc

    return GenerateResponse(
        id=result["id"],
        model=result["model"],
        content=result["content"],
        usage=UsageInfo(**result["usage"]),
        cost=result["cost"],
        latency_ms=result["latency_ms"],
        cached=result.get("cached", False),
    )


@router.post(
    "/generate/stream",
    response_class=StreamingResponse,
    summary="Stream a text completion (Server-Sent Events)",
    operation_id="llm_generate_stream",
)
async def generate_stream(
    body: GenerateRequest,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    """Stream an LLM completion as Server-Sent Events."""
    user_id = _get_user_id(request)

    async def _event_generator():
        try:
            async for chunk in service.generate_stream(
                prompt=body.prompt,
                model=body.model,
                user_id=user_id,
            ):
                data = json.dumps({"delta": chunk, "finish_reason": None})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except ValueError:
            err = json.dumps({"error": "Invalid request"})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── REST: Embed ─────────────────────────────────────────────────────────────

@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Generate text embeddings",
    operation_id="llm_embed",
)
async def generate_embeddings(
    body: EmbedRequest,
    service: LLMService = Depends(get_llm_service),
) -> EmbedResponse:
    """Generate vector embeddings for text or a list of texts."""
    texts = [body.input] if isinstance(body.input, str) else body.input
    embeddings = service.embed(texts, model=body.model)
    total_tokens = sum(len(t) // 4 for t in texts)
    return EmbedResponse(
        model=body.model,
        embeddings=embeddings,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ─── REST: Chat ───────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Chat completion",
    operation_id="llm_chat",
)
async def chat_completion(
    body: ChatRequest,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    """Chat completion with a messages list. Supports GPT-style message arrays."""
    user_id = _get_user_id(request)
    try:
        result = await service.chat(
            messages=[m.model_dump() for m in body.messages],
            model=body.model,
            user_id=user_id,
            idempotency_key=body.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid request") from exc

    return ChatResponse(
        id=result["id"],
        model=result["model"],
        message=SchemaChatMessage(role="assistant", content=result["content"]),
        usage=UsageInfo(**result["usage"]),
        cost=result["cost"],
        latency_ms=result["latency_ms"],
    )


# ─── REST: RAG Query ──────────────────────────────────────────────────────────

@router.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    summary="RAG-augmented query",
    operation_id="llm_rag_query",
)
async def rag_query(
    body: RAGQueryRequest,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> RAGQueryResponse:
    """Retrieve relevant documents then generate a grounded answer."""
    user_id = _get_user_id(request)
    result = await service.rag_query(
        query=body.query,
        top_k=body.top_k,
        model=body.model,
        user_id=user_id,
    )
    return RAGQueryResponse(
        id=result["id"],
        query=result["query"],
        answer=result["answer"],
        documents=[RAGDocument(**d) for d in result["documents"]],
        usage=UsageInfo(**result["usage"]),
        cost=result.get("cost", 0.0),
        latency_ms=result.get("latency_ms", 0.0),
    )


# ─── REST: Models list ────────────────────────────────────────────────────────

@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available LLM models",
    operation_id="llm_list_models",
)
async def list_models(
    service: LLMService = Depends(get_llm_service),
) -> ModelsListResponse:
    """Return all available LLM model definitions with pricing and capabilities."""
    models = service.list_models()
    return ModelsListResponse(
        models=[ModelInfo(**m) for m in models],
        total=len(models),
    )


# ─── REST: Cost usage ────────────────────────────────────────────────────────

@router.get(
    "/cost/usage",
    response_model=CostUsageResponse,
    summary="Get LLM cost usage for the current user",
    operation_id="llm_cost_usage",
)
async def cost_usage(
    request: Request,
    period: str | None = Query(None, description="e.g. '2026-07'"),
    service: LLMService = Depends(get_llm_service),
) -> CostUsageResponse:
    """Return cost and token usage summary for the authenticated user."""
    user_id = _get_user_id(request) or "anonymous"
    report = service.cost_usage(user_id=user_id, period=period)
    return CostUsageResponse(
        user_id=user_id,
        period=report.get("period", "all-time"),
        total_requests=report.get("total_requests", 0),
        total_tokens=report.get("total_tokens", 0),
        total_cost_usd=report.get("total_cost_usd", 0.0),
        cost_by_model=report.get("cost_by_model", {}),
        cost_by_day=report.get("cost_by_day", []),
    )


# ─── WebSocket: Streaming chat ───────────────────────────────────────────────

@router.websocket("/chat/ws")
async def websocket_chat(
    websocket: WebSocket,
    service: LLMService = Depends(get_llm_service),
) -> None:
    """Streaming chat over WebSocket.

    Client sends: ``{"messages": [...], "model": "gpt-4-turbo"}``
    Server streams: ``{"delta": "...", "finish_reason": null}`` chunks
    then: ``{"delta": "", "finish_reason": "stop"}``
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            messages = data.get("messages", [])
            model = data.get("model", "gpt-4-turbo")
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )

            try:
                async for chunk in service.generate_stream(
                    prompt=last_user, model=model
                ):
                    await websocket.send_json({"delta": chunk, "finish_reason": None})
                await websocket.send_json({"delta": "", "finish_reason": "stop"})
            except ValueError:
                await websocket.send_json({"error": "Invalid request"})

    except WebSocketDisconnect:
        logger.debug("WebSocket chat client disconnected")


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    service: LLMService = Depends(get_llm_service),
) -> None:
    """Generic streaming WebSocket endpoint.

    Client sends: ``{"prompt": "...", "model": "gpt-4-turbo"}``
    Server streams token chunks then sends finish marker.
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            prompt = data.get("prompt", "")
            model = data.get("model", "gpt-4-turbo")
            try:
                async for chunk in service.generate_stream(prompt=prompt, model=model):
                    await websocket.send_json({"delta": chunk, "finish_reason": None})
                await websocket.send_json({"delta": "", "finish_reason": "stop"})
            except ValueError:
                await websocket.send_json({"error": "Invalid request"})
    except WebSocketDisconnect:
        logger.debug("WebSocket stream client disconnected")
import hashlib
import os
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, TypeVar, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import AuthContext, get_current_auth
from api.database import get_db
from api.models.orm import LLMFeedback
from api.schemas import (
    CostDashboardResponse,
    LLMFeedbackDashboard,
    LLMFeedbackIn,
    LLMFeedbackOut,
    LLMFeedbackTrend,
    LLMPromptImprovement,
    SearchRequest,
    SearchResponse,
    SuggestionResponse,
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse,
    SupportedLanguagesResponse,
    LocaleFormatRequest,
    LocaleFormatResponse,
    TranslationCacheStatsResponse,
)
from api.services.llm_context import MultiModalContextHandler
from api.services.llm_cost import CostMonitoringService
from api.services.llm_explainer import TransactionExplainer
from api.services.llm_query import QueryTranslator
from api.services.llm_rag import build_citations, build_rag_answer, retrieve_sources
from api.services.llm_search import SemanticSearchService
from api.services.llm_suggest import AutocompleteService
from api.services.llm_validation import ResponseValidator
from api.services.translation import translation_service
from astroml.llm.compliance_logger import compliance_logger
from astroml.llm.embedding_cache import EmbeddingCache
from astroml.llm.embedding_drift import EmbeddingDriftMonitor
from astroml.llm.memory import ConversationMemory
from astroml.llm.provider import MockLLMProvider
from astroml.llm.providers.embedding_router import build_default_router

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])
explainer = TransactionExplainer()
query_translator = QueryTranslator()
context_handler = MultiModalContextHandler()
validator = ResponseValidator()
memory = ConversationMemory()
llm_provider = MockLLMProvider()
embedding_cache = EmbeddingCache()
embedding_router = build_default_router()
suggest_service = AutocompleteService()
search_service = SemanticSearchService()
cost_service = CostMonitoringService()

# Drift monitor — dimension inferred lazily from first observed vector.
# Default to 384 (HuggingFace MiniLM-L6-v2 fallback dim); reconfigured at
# runtime if the active provider returns a different dimension.
_DRIFT_MONITOR_DIM = int(os.getenv("EMBEDDING_DRIFT_DIM", "384"))
drift_monitor = EmbeddingDriftMonitor(
    n_dims=_DRIFT_MONITOR_DIM,
    provider_name="default",
    check_every=50,
)


async def log_llm_interaction(
    db: AsyncSession,
    feature: str,
    prompt: str,
    response: str,
    interaction_type: str = "query",
    auth: AuthContext = None,
    request: Request = None,
    status: str = "success",
    error_message: str = None,
    tokens_used: int = None,
    latency_ms: int = None,
) -> None:
    """Log an LLM interaction with compliance and audit trail."""
    try:
        user_id = None
        username = None
        ip_address = None
        user_agent = None

        if auth:
            user_id = getattr(auth, "user_id", None)
            username = getattr(auth, "username", None)

        if request:
            user_agent = request.headers.get("user-agent")
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                ip_address = forwarded_for.split(",")[0].strip()
            elif request.client:
                ip_address = request.client.host

        await compliance_logger.log_interaction(
            db,
            user_id=user_id,
            username=username,
            interaction_type=interaction_type,
            feature=feature,
            prompt=prompt,
            response=response,
            status=status,
            error_message=error_message,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    except Exception:
        pass




class ExplainRequest(BaseModel):
    tx_details: str

class ExplainResponse(BaseModel):
    explanation: str

@router.get("/suggest", response_model=SuggestionResponse)
async def suggest_query(q: str, max_results: int = 5, auth: AuthContext = Depends(get_current_auth)):
    try:
        return suggest_service.suggest(q, max_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest, auth: AuthContext = Depends(get_current_auth)):
    try:
        return await search_service.search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/costs/dashboard", response_model=CostDashboardResponse)
async def get_cost_dashboard(auth: AuthContext = Depends(get_current_auth)):
    try:
        return cost_service.get_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/explain", response_model=ExplainResponse)
async def explain_transaction(
    request: ExplainRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    start_time = time.time()
    try:
        explanation = await explainer.explain(request.tx_details)
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="explain",
            prompt=request.tx_details,
            response=explanation,
            interaction_type="explain",
            auth=auth,
            request=http_request,
            latency_ms=latency_ms,
        )
        return ExplainResponse(explanation=explanation)
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="explain",
            prompt=request.tx_details,
            response="",
            interaction_type="explain",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=500, detail="Internal server error")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    sql: str

@router.post("/query", response_model=QueryResponse)
async def translate_query(
    request: QueryRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    start_time = time.time()
    try:
        sql = query_translator.translate_to_sql(request.query)
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="query_translation",
            prompt=request.query,
            response=sql,
            interaction_type="translate",
            auth=auth,
            request=http_request,
            latency_ms=latency_ms,
        )
        return QueryResponse(sql=sql)
    except ValueError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="query_translation",
            prompt=request.query,
            response="",
            interaction_type="translate",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="query_translation",
            prompt=request.query,
            response="",
            interaction_type="translate",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=500, detail="Internal server error")

class ContextRequest(BaseModel):
    edges: List[Dict[str, Any]] = []
    data_points: List[float] = []

class ContextResponse(BaseModel):
    graph_summary: str
    time_series_trend: str
    mermaid: str

@router.post("/context", response_model=ContextResponse)
async def get_multimodal_context(
    request: ContextRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    start_time = time.time()
    try:
        summary = context_handler.serialize_and_summarize_graph(request.edges)
        trend = context_handler.extract_time_series(request.data_points)
        mermaid = context_handler.generate_mermaid_diagram([], request.edges)
        latency_ms = int((time.time() - start_time) * 1000)
        context_str = f"edges: {len(request.edges)}, data_points: {len(request.data_points)}"
        await log_llm_interaction(
            db,
            feature="context",
            prompt=context_str,
            response=mermaid,
            interaction_type="context",
            auth=auth,
            request=http_request,
            latency_ms=latency_ms,
        )
        return ContextResponse(
            graph_summary=summary,
            time_series_trend=trend,
            mermaid=mermaid
        )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        context_str = f"edges: {len(request.edges)}, data_points: {len(request.data_points)}"
        await log_llm_interaction(
            db,
            feature="context",
            prompt=context_str,
            response="",
            interaction_type="context",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=500, detail="Internal server error")

class ValidateRequest(BaseModel):
    raw_response: Dict[str, Any]
    context: str

class ValidateResponse(BaseModel):
    validated_response: Dict[str, Any]

@router.post("/validate", response_model=ValidateResponse)
async def validate_response(
    request: ValidateRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    start_time = time.time()
    try:
        validated = validator.validate_and_guard(request.raw_response, request.context)
        latency_ms = int((time.time() - start_time) * 1000)
        import json
        response_str = json.dumps(validated)
        await log_llm_interaction(
            db,
            feature="validate",
            prompt=request.context,
            response=response_str,
            interaction_type="validate",
            auth=auth,
            request=http_request,
            latency_ms=latency_ms,
        )
        return ValidateResponse(validated_response=validated)
    except ValueError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="validate",
            prompt=request.context,
            response="",
            interaction_type="validate",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="validate",
            prompt=request.context,
            response="",
            interaction_type="validate",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


class AskRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    source_id: str
    title: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    mode: str


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    start_time = time.time()
    try:
        sources = retrieve_sources(request.question)
        citations = build_citations(request.question, sources)
        answer = build_rag_answer(request.question, citations)
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="ask",
            prompt=request.question,
            response=answer,
            interaction_type="ask",
            auth=auth,
            request=http_request,
            latency_ms=latency_ms,
        )
        return AskResponse(
            answer=answer,
            citations=[CitationResponse(**citation.__dict__) for citation in citations],
            mode="mock-rag",
        )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_llm_interaction(
            db,
            feature="ask",
            prompt=request.question,
            response="",
            interaction_type="ask",
            auth=auth,
            request=http_request,
            status="error",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


class StreamRequest(BaseModel):
    prompt: str


async def generate_stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Example streaming response generator."""
    response_chunks = [
        "This is",
        " a streaming",
        " response",
        " from the",
        " LLM service."
    ]
    for chunk in response_chunks:
        yield chunk + "\n"
        import asyncio
        await asyncio.sleep(0.1)


@router.post("/stream")
async def stream_response(
    request: StreamRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    """Streaming endpoint for LLM responses."""
    async def logged_stream_response(prompt: str) -> AsyncGenerator[str, None]:
        start_time = time.time()
        try:
            response_parts = []
            async for chunk in generate_stream_response(prompt):
                response_parts.append(chunk)
                yield chunk
            latency_ms = int((time.time() - start_time) * 1000)
            await log_llm_interaction(
                db,
                feature="stream",
                prompt=prompt,
                response="".join(response_parts),
                interaction_type="stream",
                auth=auth,
                request=http_request,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            await log_llm_interaction(
                db,
                feature="stream",
                prompt=prompt,
                response="",
                interaction_type="stream",
                auth=auth,
                request=http_request,
                status="error",
                error_message=str(e),
                latency_ms=latency_ms,
            )
            raise

    return StreamingResponse(
        logged_stream_response(request.prompt),
        media_type="text/plain"
    )


# Feedback collection for LLM outputs (#402)
@router.post("/feedback", response_model=LLMFeedbackOut, status_code=201)
async def submit_llm_feedback(
    payload: LLMFeedbackIn,
    db: AsyncSession = Depends(get_db),
) -> LLMFeedback:
    """Collect one-click/user or weighted expert feedback for an LLM output."""
    weight = payload.expert_weight if payload.is_expert else 1.0
    feedback = LLMFeedback(
        feature=payload.feature,
        prompt=payload.prompt,
        output=payload.output,
        rating=payload.rating,
        comment=payload.comment,
        user_id=payload.user_id,
        is_expert=payload.is_expert,
        expert_weight=weight,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback


@router.get("/feedback/dashboard", response_model=LLMFeedbackDashboard)
async def llm_feedback_dashboard(db: AsyncSession = Depends(get_db)) -> LLMFeedbackDashboard:
    """Return trend metrics used by the LLM feedback dashboard."""
    rows = (await db.execute(select(LLMFeedback))).scalars().all()
    grouped: dict[str, list[LLMFeedback]] = {}
    for row in rows:
        grouped.setdefault(row.feature, []).append(row)

    trends = []
    for feature, items in sorted(grouped.items()):
        count = len(items)
        avg = sum(item.rating for item in items) / count
        weight_total = sum(item.expert_weight for item in items)
        weighted = sum(item.rating * item.expert_weight for item in items) / weight_total
        trends.append(
            LLMFeedbackTrend(
                feature=feature,
                count=count,
                average_rating=round(avg, 2),
                weighted_average_rating=round(weighted, 2),
                expert_count=sum(1 for item in items if item.is_expert),
            )
        )

    low_examples = sorted(rows, key=lambda item: (item.rating, -item.id))[:5]
    return LLMFeedbackDashboard(
        total=len(rows),
        trends=trends,
        low_rating_examples=[LLMFeedbackOut.model_validate(item) for item in low_examples],
    )


@router.get("/feedback/prompt-improvements", response_model=list[LLMPromptImprovement])
async def llm_prompt_improvements(db: AsyncSession = Depends(get_db)) -> list[LLMPromptImprovement]:
    """Summarize feedback into prompt-improvement recommendations."""
    low_rows = (
        await db.execute(select(LLMFeedback).where(LLMFeedback.rating <= 3))
    ).scalars().all()
    by_feature: dict[str, list[LLMFeedback]] = {}
    for row in low_rows:
        by_feature.setdefault(row.feature, []).append(row)

    return [
        LLMPromptImprovement(
            feature=feature,
            evidence_count=len(items),
            recommendation=(
                "Revise the prompt to request concise, cited, schema-valid output; "
                "prioritize expert comments when available."
            ),
        )
        for feature, items in sorted(by_feature.items())
    ]


# ─── Translation endpoints (Issue 1) ────────────────────────────────────────

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    target_language: str = Field(..., min_length=2, max_length=10)
    source_language: Optional[str] = Field(default=None, min_length=2, max_length=10)
    use_cache: bool = True


class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    cached: bool
    latency_ms: float


class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    target_language: str = Field(..., min_length=2, max_length=10)
    source_language: Optional[str] = Field(default=None, min_length=2, max_length=10)
    use_cache: bool = True


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]
    total_latency_ms: float


class SupportedLanguagesResponse(BaseModel):
    languages: Dict[str, Dict[str, str]]


@router.get("/translate/languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages(auth: AuthContext = Depends(get_current_auth)):
    """Get list of supported languages for translation."""
    return SupportedLanguagesResponse(languages=translation_service.get_supported_languages())


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    auth: AuthContext = Depends(get_current_auth),
):
    """Translate text to target language."""
    try:
        result = await translation_service.translate(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language,
            use_cache=request.use_cache,
        )
        return TranslationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(
    request: BatchTranslationRequest,
    auth: AuthContext = Depends(get_current_auth),
):
    """Translate multiple texts to target language."""
    try:
        results = await translation_service.translate_batch(
            texts=request.texts,
            target_language=request.target_language,
            source_language=request.source_language,
            use_cache=request.use_cache,
        )
        total_latency = sum(r["latency_ms"] for r in results)
        return BatchTranslationResponse(
            translations=[TranslationResponse(**r) for r in results],
            total_latency_ms=total_latency,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


class LocaleFormatRequest(BaseModel):
    value: Union[float, int, str]
    locale: str = Field(..., min_length=2, max_length=10)
    format_type: str = Field(..., pattern="^(number|currency|percent|date|datetime)$")
    currency_code: Optional[str] = Field(default=None, min_length=3, max_length=3)


class LocaleFormatResponse(BaseModel):
    formatted: str
    locale: str
    format_type: str


@router.post("/translate/format", response_model=LocaleFormatResponse)
async def format_locale(
    request: LocaleFormatRequest,
    auth: AuthContext = Depends(get_current_auth),
):
    """Format numbers, currencies, dates, etc. for a specific locale."""
    try:
        formatted = translation_service.format_locale(
            value=request.value,
            locale=request.locale,
            format_type=request.format_type,
            currency_code=request.currency_code,
        )
        return LocaleFormatResponse(
            formatted=formatted,
            locale=request.locale,
            format_type=request.format_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


class TranslationCacheStatsResponse(BaseModel):
    hits: int
    misses: int
    sets: int
    invalidations: int
    hit_rate: float
    size: int


@router.get("/translate/cache/stats", response_model=TranslationCacheStatsResponse)
async def get_translation_cache_stats(auth: AuthContext = Depends(get_current_auth)):
    """Get translation cache statistics."""
    stats = translation_service.get_cache_stats()
    return TranslationCacheStatsResponse(**stats)


@router.post("/translate/cache/invalidate")
async def invalidate_translation_cache(
    text: Optional[str] = None,
    auth: AuthContext = Depends(get_current_auth),
):
    """Invalidate translation cache (specific text or all)."""
    if text:
        translation_service.invalidate_cache(text)
        return {"message": "Cache entry invalidated", "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16]}
    else:
        count = translation_service.invalidate_all_cache()
        return {"message": f"Invalidated {count} cache entries"}
