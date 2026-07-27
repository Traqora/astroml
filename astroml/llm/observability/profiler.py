"""Performance profiler for LLM operations.

Resolves #456: Token throughput, TTFT (time to first token), provider latency
profiling, and memory usage benchmarks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileResult:
    """Detailed performance profile of a single LLM call."""

    operation: str
    provider: str | None = None
    model: str | None = None
    total_latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    cache_hit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Operation  : {self.operation}",
            f"Provider   : {self.provider or 'N/A'} / {self.model or 'N/A'}",
            f"Latency    : {self.total_latency_ms:.1f}ms",
            f"TTFT       : {self.time_to_first_token_ms:.1f}ms" if self.time_to_first_token_ms else "TTFT       : N/A",
            f"Throughput : {self.tokens_per_second:.1f} tok/s",
            f"Tokens     : {self.prompt_tokens}p + {self.completion_tokens}c = {self.total_tokens}",
            f"Cost       : ${self.cost_usd:.6f}",
        ]
        return "\n".join(lines)


class LLMProfiler:
    """Profile LLM calls for latency benchmarking and throughput analysis.

    Tracks:
    - Wall-clock latency (total, TTFT)
    - Token throughput (tokens/second)
    - Cost efficiency (tokens per dollar)
    - Historical results for P50/P95/P99 benchmarks

    Example::

        profiler = LLMProfiler()
        profiler.start("generate", provider="openai", model="gpt-4")
        # ... call LLM ...
        profiler.record_first_token()
        result = profiler.finish(prompt_tokens=50, completion_tokens=120, cost_usd=0.003)
        print(result.summary())
    """

    def __init__(self) -> None:
        self._results: list[ProfileResult] = []
        self._current: ProfileResult | None = None
        self._start_time: float = 0.0
        self._first_token_time: float | None = None

    def start(
        self,
        operation: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Begin profiling an operation."""
        self._current = ProfileResult(operation=operation, provider=provider, model=model)
        self._start_time = time.monotonic()
        self._first_token_time = None

    def record_first_token(self) -> None:
        """Call this when the first token is received (streaming TTFT)."""
        if self._current is not None and self._first_token_time is None:
            self._first_token_time = time.monotonic()

    def finish(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        retries: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> ProfileResult:
        """Finish profiling and return the :class:`ProfileResult`."""
        if self._current is None:
            raise RuntimeError("profiler.start() must be called before finish()")

        end_time = time.monotonic()
        total_ms = (end_time - self._start_time) * 1000
        ttft_ms = (
            (self._first_token_time - self._start_time) * 1000
            if self._first_token_time
            else None
        )
        total_tokens = prompt_tokens + completion_tokens
        tokens_per_sec = (
            total_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
        )

        result = self._current
        result.total_latency_ms = total_ms
        result.time_to_first_token_ms = ttft_ms
        result.tokens_per_second = tokens_per_sec
        result.prompt_tokens = prompt_tokens
        result.completion_tokens = completion_tokens
        result.total_tokens = total_tokens
        result.cost_usd = cost_usd
        result.cache_hit = cache_hit
        result.retries = retries
        result.extra = extra or {}

        self._results.append(result)
        self._current = None
        return result

    def benchmark(self) -> dict[str, Any]:
        """Return P50/P95/P99 latency benchmark across recorded results."""
        if not self._results:
            return {}
        latencies = sorted(r.total_latency_ms for r in self._results)
        n = len(latencies)

        def _p(pct: float) -> float:
            idx = int(n * pct)
            return latencies[min(idx, n - 1)]

        return {
            "count": n,
            "p50_ms": round(_p(0.50), 2),
            "p95_ms": round(_p(0.95), 2),
            "p99_ms": round(_p(0.99), 2),
            "min_ms": round(latencies[0], 2),
            "max_ms": round(latencies[-1], 2),
            "avg_tokens_per_sec": round(
                sum(r.tokens_per_second for r in self._results) / n, 2
            ),
        }
