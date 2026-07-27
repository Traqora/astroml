"""Tests for mock LLM providers — unit tests for the test infrastructure itself.

Resolves #458: Validates deterministic behaviour, error injection,
streaming simulation, and token counting of mock providers.
"""
from __future__ import annotations

import pytest
import asyncio

from tests.llm.mocks import DeterministicMockProvider, ErrorInjectingProvider, HighLatencyProvider


class TestDeterministicMockProvider:
    """Unit tests for DeterministicMockProvider."""

    def test_deterministic_same_prompt_same_response(self, mock_provider):
        """Same prompt must always return the same response."""
        r1 = mock_provider.generate("Hello world")
        r2 = mock_provider.generate("Hello world")
        assert r1 == r2

    def test_different_prompts_different_responses(self, mock_provider):
        """Different prompts should produce different responses."""
        r1 = mock_provider.generate("Hello world")
        r2 = mock_provider.generate("Goodbye world")
        assert r1 != r2

    def test_custom_response_override(self):
        """Custom response mappings should override hash-based responses."""
        provider = DeterministicMockProvider(
            custom_responses={"exact prompt": "exact response"}
        )
        assert provider.generate("exact prompt") == "exact response"

    def test_returns_string(self, mock_provider):
        """generate() must return a non-empty string."""
        result = mock_provider.generate("test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_token_counting(self, mock_provider):
        """Token counting should follow the 4-chars-per-token heuristic."""
        tokens = mock_provider.count_tokens("Hello world!")
        assert tokens == max(1, len("Hello world!") // 4)

    def test_cost_estimation(self, mock_provider):
        """Cost estimation should return a non-negative float."""
        cost = mock_provider.estimate_cost("short prompt", "short response")
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_error_injection_on_specified_calls(self):
        """Error injection should raise on the specified call indices."""
        provider = DeterministicMockProvider(error_on_calls=[0, 2])
        with pytest.raises(RuntimeError):
            provider.generate("test")
        # Call 1 (index 1) should succeed
        result = provider.generate("test")
        assert isinstance(result, str)
        # Call 2 (index 2) should error
        with pytest.raises(RuntimeError):
            provider.generate("test")

    def test_reset_clears_call_count(self):
        """reset() should allow error injection to restart."""
        provider = DeterministicMockProvider(error_on_calls=[0])
        with pytest.raises(RuntimeError):
            provider.generate("test")
        provider.reset()
        # After reset, call 0 should error again
        with pytest.raises(RuntimeError):
            provider.generate("test")

    @pytest.mark.asyncio
    async def test_streaming_yields_chunks(self, mock_provider):
        """Streaming should yield at least one non-empty chunk."""
        stream = await mock_provider.generate_stream("Hello")
        chunks = []
        async for chunk in stream.get_chunks():
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    @pytest.mark.asyncio
    async def test_streaming_reconstructs_full_response(self, mock_provider):
        """Streaming chunks concatenated should equal the generate() output."""
        prompt = "test streaming"
        expected = mock_provider.generate(prompt)
        mock_provider.reset()

        # Rebuild from stream
        stream = await mock_provider.generate_stream(prompt)
        chunks = []
        async for chunk in stream.get_chunks():
            chunks.append(chunk)
        reconstructed = "".join(chunks)
        assert reconstructed == expected


class TestErrorInjectingProvider:
    """Tests for error injection."""

    def test_always_raises(self):
        provider = ErrorInjectingProvider()
        with pytest.raises(RuntimeError):
            provider.generate("test")

    def test_custom_exception(self):
        exc = ValueError("custom error")
        provider = ErrorInjectingProvider(exception=exc)
        with pytest.raises(ValueError, match="custom error"):
            provider.generate("test")

    @pytest.mark.asyncio
    async def test_stream_also_raises(self):
        provider = ErrorInjectingProvider()
        with pytest.raises(RuntimeError):
            await provider.generate_stream("test")


class TestHighLatencyProvider:
    """Tests for high-latency provider."""

    def test_produces_valid_response(self):
        """High-latency provider should still return valid responses."""
        provider = HighLatencyProvider(latency_ms=0.1)  # minimal latency for test speed
        result = provider.generate("test")
        assert isinstance(result, str)
