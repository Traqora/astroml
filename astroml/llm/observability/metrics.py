"""Prometheus metrics collection for LLM operations.

Resolves #456: Exposes requests/min, p50/p95/p99 latency, token throughput,
error rates, and cache hit rates as Prometheus metrics.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

# Optional prometheus_client — gracefully degrade if not installed
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False


class _InMemoryHistogram:
    """Simple in-memory histogram for environments without prometheus_client."""

    def __init__(self, window: int = 3600) -> None:
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, float]] = deque()  # (timestamp, value)
        self._window = window  # seconds

    def observe(self, value: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, value))
            # Evict samples outside the window
            while self._samples and self._samples[0][0] < now - self._window:
                self._samples.popleft()

    def percentile(self, p: float) -> float:
        with self._lock:
            vals = sorted(v for _, v in self._samples)
        if not vals:
            return 0.0
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

    def count(self) -> int:
        with self._lock:
            return len(self._samples)


class LLMMetrics:
    """Metrics collection hub for all LLM operations.

    Wraps Prometheus counters/histograms when available, falling back to
    lightweight in-memory structures for testing and local development.

    Prometheus metrics exposed:
    - ``llm_requests_total`` — Counter by operation, provider, model, status
    - ``llm_request_latency_seconds`` — Histogram of response latencies
    - ``llm_tokens_total`` — Counter by type (prompt/completion)
    - ``llm_cost_total_usd`` — Counter of LLM costs
    - ``llm_cache_hits_total`` — Counter of cache hits
    - ``llm_errors_total`` — Counter by provider and error type

    Example::

        metrics = LLMMetrics()
        metrics.record_request("generate", provider="openai", model="gpt-4", status="success")
        metrics.record_latency("generate", latency_ms=820)
        metrics.record_tokens(prompt=50, completion=120)
        p95 = metrics.latency_percentile(95)
    """

    def __init__(self, registry: Any = None) -> None:
        self._latency_hist = _InMemoryHistogram()
        self._request_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._token_counts = {"prompt": 0, "completion": 0}
        self._cost_total: float = 0.0
        self._cache_hits: int = 0
        self._lock = threading.Lock()

        # Prometheus metrics (optional)
        if _PROMETHEUS_AVAILABLE:
            reg = registry or CollectorRegistry()
            self._prom_requests = Counter(
                "llm_requests_total",
                "Total LLM requests",
                ["operation", "provider", "model", "status"],
                registry=reg,
            )
            self._prom_latency = Histogram(
                "llm_request_latency_seconds",
                "LLM request latency in seconds",
                ["operation", "provider"],
                buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
                registry=reg,
            )
            self._prom_tokens = Counter(
                "llm_tokens_total",
                "Total LLM tokens used",
                ["token_type"],
                registry=reg,
            )
            self._prom_cost = Counter(
                "llm_cost_total_usd",
                "Total LLM cost in USD",
                registry=reg,
            )
            self._prom_cache = Counter(
                "llm_cache_hits_total",
                "Total cache hits",
                registry=reg,
            )
            self._prom_errors = Counter(
                "llm_errors_total",
                "Total LLM errors",
                ["provider", "error_type"],
                registry=reg,
            )
        else:
            self._prom_requests = None

    def record_request(
        self,
        operation: str,
        provider: str = "unknown",
        model: str = "unknown",
        status: str = "success",
    ) -> None:
        key = f"{operation}:{provider}:{model}:{status}"
        with self._lock:
            self._request_counts[key] = self._request_counts.get(key, 0) + 1
        if self._prom_requests:
            self._prom_requests.labels(operation, provider, model, status).inc()

    def record_latency(self, operation: str, latency_ms: float, provider: str = "unknown") -> None:
        self._latency_hist.observe(latency_ms)
        if self._prom_requests:
            self._prom_latency.labels(operation, provider).observe(latency_ms / 1000)

    def record_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        with self._lock:
            self._token_counts["prompt"] += prompt
            self._token_counts["completion"] += completion
        if self._prom_requests:
            self._prom_tokens.labels("prompt").inc(prompt)
            self._prom_tokens.labels("completion").inc(completion)

    def record_cost(self, cost_usd: float) -> None:
        with self._lock:
            self._cost_total += cost_usd
        if self._prom_requests:
            self._prom_cost.inc(cost_usd)

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1
        if self._prom_requests:
            self._prom_cache.inc()

    def record_error(self, provider: str = "unknown", error_type: str = "unknown") -> None:
        key = f"{provider}:{error_type}"
        with self._lock:
            self._error_counts[key] = self._error_counts.get(key, 0) + 1
        if self._prom_requests:
            self._prom_errors.labels(provider, error_type).inc()

    def latency_percentile(self, p: float) -> float:
        """Return the p-th percentile latency in milliseconds."""
        return self._latency_hist.percentile(p)

    def snapshot(self) -> dict[str, Any]:
        """Return a metrics snapshot for dashboards and health checks."""
        with self._lock:
            return {
                "request_counts": dict(self._request_counts),
                "error_counts": dict(self._error_counts),
                "token_counts": dict(self._token_counts),
                "cost_total_usd": round(self._cost_total, 6),
                "cache_hits": self._cache_hits,
                "latency_p50_ms": round(self.latency_percentile(50), 2),
                "latency_p95_ms": round(self.latency_percentile(95), 2),
                "latency_p99_ms": round(self.latency_percentile(99), 2),
                "total_samples": self._latency_hist.count(),
            }
