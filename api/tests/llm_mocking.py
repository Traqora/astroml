"""Deterministic LLM mocking helpers for integration tests (#401)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class LLMCall:
    prompt: str
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class DeterministicLLMMock:
    responses: dict[str, str] = field(default_factory=dict)
    default_response: str = "mock llm response"
    delay_ms: float = 0.0
    fail_for: set[str] = field(default_factory=set)
    calls: list[LLMCall] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        if any(marker in prompt for marker in self.fail_for):
            raise RuntimeError("injected LLM failure")
        response = self.responses.get(prompt, self.default_response)
        latency_ms = (time.perf_counter() - start) * 1000
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(response) // 4)
        cost_usd = (prompt_tokens * 0.00001) + (completion_tokens * 0.00002)
        self.calls.append(LLMCall(prompt, response, latency_ms, prompt_tokens, completion_tokens, cost_usd))
        return response

    async def generate_stream(self, prompt: str) -> Iterable[str]:
        response = self.generate(prompt)
        for token in response.split():
            await asyncio.sleep(0)
            yield token

    def p95_latency_ms(self) -> float:
        values = sorted(call.latency_ms for call in self.calls)
        if not values:
            return 0.0
        index = min(len(values) - 1, int(0.95 * (len(values) - 1)))
        return values[index]

    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)
