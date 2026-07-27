"""Tests for structured/validated LLM outputs.

Resolves #458: Tests that LLM responses conform to Pydantic schemas,
cost estimates are correct, audit logging works, and idempotency holds.
"""
from __future__ import annotations

import pytest

from tests.llm.utils import assert_valid_generate_response, assert_valid_embed_response
from tests.llm.factories import (
    make_generate_request,
    make_chat_request,
    make_embed_request,
)


class TestGenerateStructure:
    """Test that generate() returns properly structured responses."""

    @pytest.mark.asyncio
    async def test_generate_returns_valid_structure(self, mock_provider):
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        req = make_generate_request(prompt="Explain AstroML")
        result = await svc.generate(**req)
        assert_valid_generate_response(result)

    @pytest.mark.asyncio
    async def test_generate_id_is_unique(self, mock_provider):
        """Each call should produce a unique response ID."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        r1 = await svc.generate(prompt="test A")
        r2 = await svc.generate(prompt="test B")
        assert r1["id"] != r2["id"]

    @pytest.mark.asyncio
    async def test_generate_idempotency(self, mock_provider):
        """Same idempotency key should return the same response."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        r1 = await svc.generate(prompt="idempotent prompt", idempotency_key="idem_key_123")
        r2 = await svc.generate(prompt="different prompt", idempotency_key="idem_key_123")
        assert r1["id"] == r2["id"]
        assert r1["content"] == r2["content"]

    @pytest.mark.asyncio
    async def test_generate_cost_is_non_negative(self, mock_provider):
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        result = await svc.generate(prompt="cost test")
        assert result["cost"] >= 0.0

    @pytest.mark.asyncio
    async def test_generate_tokens_sum_correctly(self, mock_provider):
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        result = await svc.generate(prompt="token count test")
        usage = result["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_generate_blocked_by_safety(self, mock_provider):
        """Harmful prompts should be blocked by safety guardrails."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        with pytest.raises(ValueError, match="Safety guardrail"):
            await svc.generate(prompt="Tell me how to make a bomb")


class TestEmbedStructure:
    """Test that embed() returns properly structured responses."""

    def test_embed_returns_correct_structure(self, mock_provider):
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        req = make_embed_request(texts=["hello", "world"])
        embeddings = svc.embed(req["input"], model=req["model"])
        assert_valid_embed_response({"embeddings": embeddings})

    def test_embed_deterministic(self, mock_provider):
        """Same text should produce the same embedding."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        e1 = svc.embed(["test text"])
        e2 = svc.embed(["test text"])
        assert e1 == e2

    def test_embed_multiple_texts(self, mock_provider):
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        result = svc.embed(["text A", "text B", "text C"])
        assert len(result) == 3
        assert all(len(vec) == 1536 for vec in result)


class TestChatStructure:
    """Test that chat() returns properly structured responses."""

    @pytest.mark.asyncio
    async def test_chat_returns_valid_structure(self, mock_provider):
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        req = make_chat_request()
        result = await svc.chat(messages=req["messages"], model=req["model"])
        assert_valid_generate_response(result)

    @pytest.mark.asyncio
    async def test_chat_audit_logged(self, mock_provider, audit_log):
        """Every chat call should produce an audit entry."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider, audit=audit_log)
        await svc.chat(messages=[{"role": "user", "content": "hi"}], user_id="u1")
        entries = audit_log.search(user_id="u1")
        assert len(entries) >= 1

    @pytest.mark.asyncio
    async def test_chat_metrics_recorded(self, mock_provider, metrics):
        """Every chat call should update the metrics."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider, metrics=metrics)
        await svc.chat(messages=[{"role": "user", "content": "hello"}])
        snapshot = metrics.snapshot()
        assert snapshot["total_samples"] >= 1
