"""Test data factories for LLM tests.

Resolves #458: Factory functions for generating test data structures
(prompts, conversations, documents, tool calls, etc.) for use in tests.
"""
from __future__ import annotations

import uuid
import random
from typing import Any


def make_prompt(
    text: str | None = None,
    length: int = 50,
    seed: int | None = None,
) -> str:
    """Generate a test prompt string.

    Args:
        text: If provided, return this exact text.
        length: Approximate length in words if text is not provided.
        seed: Random seed for reproducibility.
    """
    if text is not None:
        return text
    rng = random.Random(seed)
    words = [
        "transaction", "fraud", "blockchain", "account", "user", "payment",
        "verify", "detect", "analyse", "report", "summary", "explain",
    ]
    return " ".join(rng.choices(words, k=length))


def make_conversation(
    n_turns: int = 3,
    include_system: bool = True,
    seed: int | None = None,
) -> list[dict[str, str]]:
    """Generate a test conversation with *n_turns* user/assistant pairs."""
    rng = random.Random(seed)
    messages: list[dict[str, str]] = []
    if include_system:
        messages.append({"role": "system", "content": "You are a helpful financial assistant."})
    topics = ["transactions", "fraud", "account balance", "model performance", "audit"]
    for i in range(n_turns):
        topic = rng.choice(topics)
        messages.append({"role": "user", "content": f"Tell me about {topic} (turn {i+1})"})
        messages.append({"role": "assistant", "content": f"Here is information about {topic}."})
    return messages


def make_rag_document(
    doc_id: str | None = None,
    content: str | None = None,
    score: float | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build a RAG document dict for testing."""
    return {
        "doc_id": doc_id or f"doc_{uuid.uuid4().hex[:6]}",
        "content": content or "This is a test document about LLM and fraud detection.",
        "score": score if score is not None else round(random.uniform(0.7, 0.99), 2),
        "metadata": metadata or {"source": "test"},
    }


def make_generate_request(
    prompt: str | None = None,
    model: str = "gpt-4-turbo",
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a GenerateRequest-compatible dict."""
    return {
        "prompt": prompt or make_prompt(seed=42),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }


def make_chat_request(
    messages: list[dict[str, str]] | None = None,
    model: str = "gpt-4-turbo",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a ChatRequest-compatible dict."""
    return {
        "messages": messages or make_conversation(n_turns=1, seed=42),
        "model": model,
        **kwargs,
    }


def make_embed_request(
    texts: list[str] | None = None,
    model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    """Build an EmbedRequest-compatible dict."""
    return {
        "input": texts or ["test embedding text", "another embedding"],
        "model": model,
    }


def make_rag_query_request(
    query: str | None = None,
    top_k: int = 3,
    model: str = "gpt-4-turbo",
) -> dict[str, Any]:
    """Build a RAGQueryRequest-compatible dict."""
    return {
        "query": query or "What transactions occurred recently?",
        "top_k": top_k,
        "model": model,
    }


def make_safety_incident(
    text: str = "test text",
    category: str = "harmful",
    decision: str = "block",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build a mock safety incident record."""
    return {
        "incident_id": str(uuid.uuid4()),
        "text_preview": text[:200],
        "category": category,
        "decision": decision,
        "user_id": user_id or "test_user",
        "confidence": 0.9,
    }


def make_audit_entry(
    operation: str = "generate",
    user_id: str | None = None,
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
    cost_usd: float = 0.003,
    latency_ms: float = 820.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a mock LLM audit entry dict."""
    return {
        "audit_id": str(uuid.uuid4()),
        "operation": operation,
        "user_id": user_id or "test_user",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "model": "gpt-4-turbo",
        "provider": "mock",
        **kwargs,
    }
