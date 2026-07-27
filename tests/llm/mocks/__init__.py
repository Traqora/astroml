"""Mock LLM providers for testing — deterministic, no real API calls.

Resolves #458: Full mock provider with deterministic responses, latency
simulation, error injection, streaming simulation, and token counting.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import AsyncGenerator, Callable

from astroml.llm.provider import LLMProvider, StreamingResponse


class DeterministicMockProvider(LLMProvider):
    """Mock LLM provider with deterministic responses based on input hash.

    Features:
    - Deterministic: same input always produces same output
    - Configurable simulated latency (default: 0ms for fast tests)
    - Error injection for resilience testing
    - Streaming simulation with configurable chunk size
    - Token counting matching real provider behaviour
    - Cost calculation using a configurable cost table

    Example::

        provider = DeterministicMockProvider(latency_ms=50)
        response = provider.generate("Hello world")
        assert response == provider.generate("Hello world")  # deterministic
    """

    # Default cost table (USD / 1k tokens)
    DEFAULT_COST: dict[str, float] = {"prompt": 0.01, "completion": 0.03}

    def __init__(
        self,
        latency_ms: float = 0.0,
        error_on_calls: list[int] | None = None,
        custom_responses: dict[str, str] | None = None,
        chunk_delay_ms: float = 5.0,
        cost_table: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            latency_ms: Simulated latency per request in milliseconds.
            error_on_calls: List of call indices (0-based) that should raise an error.
            custom_responses: Mapping of prompt -> response for specific overrides.
            chunk_delay_ms: Delay between streaming chunks in milliseconds.
            cost_table: Override cost table {"prompt": float, "completion": float}.
        """
        self._latency_ms = latency_ms
        self._error_on_calls = set(error_on_calls or [])
        self._custom_responses = custom_responses or {}
        self._chunk_delay_ms = chunk_delay_ms
        self._cost_table = cost_table or self.DEFAULT_COST
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate a deterministic response for *prompt*."""
        self._maybe_error()
        self._simulate_latency()
        return self._deterministic_response(prompt)

    async def generate_stream(self, prompt: str) -> StreamingResponse:
        """Simulate streaming by yielding individual words with configurable delay."""
        self._maybe_error()

        response = self._deterministic_response(prompt)
        words = response.split()

        async def _generator() -> AsyncGenerator[str, None]:
            for i, word in enumerate(words):
                if self._chunk_delay_ms > 0:
                    await asyncio.sleep(self._chunk_delay_ms / 1000)
                yield word if i == len(words) - 1 else word + " "

        return StreamingResponse(_generator())

    def count_tokens(self, text: str) -> int:
        """Estimate token count using the standard ~4 chars/token heuristic."""
        return max(1, len(text) // 4)

    def estimate_cost(self, prompt: str, response: str) -> float:
        """Return estimated cost in USD for the prompt + response pair."""
        p_tokens = self.count_tokens(prompt)
        c_tokens = self.count_tokens(response)
        return (
            p_tokens / 1000 * self._cost_table["prompt"]
            + c_tokens / 1000 * self._cost_table["completion"]
        )

    def reset(self) -> None:
        """Reset call counter (useful between test cases)."""
        self._call_count = 0

    # ─── Private helpers ────────────────────────────────────────────────────

    def _deterministic_response(self, prompt: str) -> str:
        if prompt in self._custom_responses:
            return self._custom_responses[prompt]
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"Mock response for prompt [{digest}]: {prompt[:40]}..."

    def _simulate_latency(self) -> None:
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000)

    def _maybe_error(self) -> None:
        idx = self._call_count
        self._call_count += 1
        if idx in self._error_on_calls:
            raise RuntimeError(f"Injected error on call #{idx}")


class ErrorInjectingProvider(DeterministicMockProvider):
    """Provider that always raises on generate — useful for error path testing."""

    def __init__(self, exception: Exception | None = None) -> None:
        super().__init__(error_on_calls=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self._exception = exception or RuntimeError("Simulated provider failure")

    def _maybe_error(self) -> None:
        raise self._exception


class HighLatencyProvider(DeterministicMockProvider):
    """Provider with configurable high latency — useful for timeout testing."""

    def __init__(self, latency_ms: float = 2000.0) -> None:
        super().__init__(latency_ms=latency_ms)
