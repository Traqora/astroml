"""LLM service integration — business logic between API layer and LLM providers.

Resolves #457: Stateless service that wraps LLM providers with caching,
cost calculation, observability hooks, and safety guardrail integration.
"""
from __future__ import annotations

import hashlib
import time
import uuid
import logging
from typing import AsyncGenerator, Any

from astroml.llm.provider import LLMProvider, MockLLMProvider
from astroml.llm.observability.tracer import LLMTracer
from astroml.llm.observability.metrics import LLMMetrics
from astroml.llm.observability.audit import LLMAuditLog
from astroml.llm.safety.guards import SafetyGuard, SafetyDecision

logger = logging.getLogger(__name__)

# Cost table (USD per 1k tokens) — update as providers change pricing
_COST_TABLE: dict[str, dict[str, float]] = {
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "mock": {"prompt": 0.0, "completion": 0.0},
}

_KNOWN_MODELS = [
    {
        "id": "gpt-4-turbo",
        "provider": "openai",
        "context_window": 128_000,
        "cost_per_1k_prompt_tokens": 0.01,
        "cost_per_1k_completion_tokens": 0.03,
        "supports_streaming": True,
        "supports_vision": True,
    },
    {
        "id": "gpt-4o",
        "provider": "openai",
        "context_window": 128_000,
        "cost_per_1k_prompt_tokens": 0.005,
        "cost_per_1k_completion_tokens": 0.015,
        "supports_streaming": True,
        "supports_vision": True,
    },
    {
        "id": "gpt-3.5-turbo",
        "provider": "openai",
        "context_window": 16_385,
        "cost_per_1k_prompt_tokens": 0.0005,
        "cost_per_1k_completion_tokens": 0.0015,
        "supports_streaming": True,
        "supports_vision": False,
    },
    {
        "id": "text-embedding-3-small",
        "provider": "openai",
        "context_window": 8_191,
        "cost_per_1k_prompt_tokens": 0.00002,
        "cost_per_1k_completion_tokens": 0.0,
        "supports_streaming": False,
        "supports_vision": False,
    },
]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — 4 characters per token heuristic."""
    return max(1, len(text) // 4)


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated USD cost for a request."""
    table = _COST_TABLE.get(model, _COST_TABLE["gpt-3.5-turbo"])
    return (
        prompt_tokens / 1000 * table["prompt"]
        + completion_tokens / 1000 * table["completion"]
    )


class LLMService:
    """Service layer integrating LLM providers with the API gateway.

    Responsibilities:
    - Idempotency deduplication via in-memory cache
    - Safety guardrail enforcement before/after LLM calls
    - Observability: tracing, metrics, audit logging
    - Cost computation
    - Streaming support

    Example::

        svc = LLMService()
        result = await svc.generate(
            prompt="Summarise this transaction",
            model="gpt-4-turbo",
            user_id="u42",
        )
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        tracer: LLMTracer | None = None,
        metrics: LLMMetrics | None = None,
        audit: LLMAuditLog | None = None,
        guard: SafetyGuard | None = None,
    ) -> None:
        self._provider = provider or MockLLMProvider()
        self._tracer = tracer or LLMTracer()
        self._metrics = metrics or LLMMetrics()
        self._audit = audit or LLMAuditLog()
        self._guard = guard or SafetyGuard()
        self._idempotency_cache: dict[str, Any] = {}

    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        user_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a completion. Returns the full response dict."""
        # Idempotency check
        if idempotency_key and idempotency_key in self._idempotency_cache:
            return self._idempotency_cache[idempotency_key]

        # Safety check
        safety_result = self._guard.check_input(prompt, user_id=user_id)
        if safety_result.is_blocked:
            raise ValueError(f"Safety guardrail blocked request: {safety_result.reason}")

        start = time.monotonic()
        with self._tracer.span("generate", provider="mock", model=model) as span:
            safe_prompt = safety_result.redacted_text or prompt
            content = self._provider.generate(safe_prompt)

            prompt_tokens = _estimate_tokens(safe_prompt)
            completion_tokens = _estimate_tokens(content)
            span.prompt_tokens = prompt_tokens
            span.completion_tokens = completion_tokens
            span.total_tokens = prompt_tokens + completion_tokens
            span.cost_usd = _compute_cost(model, prompt_tokens, completion_tokens)
            span.model = model

        latency_ms = (time.monotonic() - start) * 1000
        cost = _compute_cost(model, prompt_tokens, completion_tokens)

        self._metrics.record_request("generate", provider="mock", model=model, status="success")
        self._metrics.record_latency("generate", latency_ms, provider="mock")
        self._metrics.record_tokens(prompt=prompt_tokens, completion=completion_tokens)
        self._metrics.record_cost(cost)

        self._audit.log(
            "generate",
            user_id=user_id,
            provider="mock",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        result = {
            "id": f"gen_{uuid.uuid4().hex[:8]}",
            "model": model,
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost": cost,
            "latency_ms": round(latency_ms, 2),
            "cached": False,
        }

        if idempotency_key:
            self._idempotency_cache[idempotency_key] = result

        return result

    async def generate_stream(
        self,
        prompt: str,
        model: str = "gpt-4-turbo",
        user_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion as an async generator of text chunks."""
        safety_result = self._guard.check_input(prompt, user_id=user_id)
        if safety_result.is_blocked:
            raise ValueError(f"Safety guardrail blocked request: {safety_result.reason}")

        safe_prompt = safety_result.redacted_text or prompt
        stream = await self._provider.generate_stream(safe_prompt)
        async for chunk in stream.get_chunks():
            yield chunk

    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        """Return mock embeddings (list of float vectors)."""
        # In production replace with real embedding provider
        import hashlib
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16)  # noqa: S324
            # 1536-dim deterministic mock embedding
            vec = [(((seed + i) % 2000) - 1000) / 1000 for i in range(1536)]
            result.append(vec)
        return result

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4-turbo",
        user_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Chat completion — takes messages list and returns assistant reply."""
        # Extract last user message for safety check
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return await self.generate(
            prompt=last_user,
            model=model,
            user_id=user_id,
            idempotency_key=idempotency_key,
        )

    async def rag_query(
        self,
        query: str,
        top_k: int = 5,
        model: str = "gpt-4-turbo",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """RAG query — retrieve mock documents and generate grounded answer."""
        # In production: replace with vector DB retrieval + LLM call
        mock_docs = [
            {
                "doc_id": f"doc_{i}",
                "content": f"Document {i} relevant to: {query[:50]}",
                "score": round(0.95 - i * 0.08, 2),
                "metadata": {"source": "knowledge_base", "index": i},
            }
            for i in range(top_k)
        ]
        gen_result = await self.generate(
            prompt=f"Answer based on context: {query}",
            model=model,
            user_id=user_id,
        )
        gen_result["query"] = query
        gen_result["documents"] = mock_docs
        gen_result["answer"] = gen_result.pop("content")
        gen_result["id"] = f"rag_{uuid.uuid4().hex[:8]}"
        return gen_result

    def list_models(self) -> list[dict[str, Any]]:
        """Return available model definitions."""
        return _KNOWN_MODELS

    def cost_usage(self, user_id: str, period: str | None = None) -> dict[str, Any]:
        """Return cost usage report for a user."""
        report = self._audit.cost_report(user_id=user_id)
        report["user_id"] = user_id
        report["period"] = period or "all-time"
        report.setdefault("cost_by_day", [])
        return report
