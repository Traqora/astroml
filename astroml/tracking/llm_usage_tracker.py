"""Token usage + cost tracking utilities for LLM calls.

This repo currently doesn't include a concrete LLM provider integration.
To keep the feature testable and useful, this module is provider-agnostic:
callers should construct an ``LLMUsage`` object from provider responses and
pass it to ``LLMUsageTracker``.

Integration points:
- Wrap your LLM provider call and record usage (tokens, latency, cost).
- Optionally register cost alerts (callbacks).
- Expose Prometheus metrics (if prometheus_client is installed).

"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover
    Counter = Gauge = Histogram = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMUsage:
    """Usage details from an LLM provider response."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # provider-calculated cost in USD (preferred)
    cost_usd: float
    # request latency seconds
    latency_s: float
    # request correlation ids (optional)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class LLMPrices:
    """Static token prices (USD per 1K tokens) for cost estimation."""

    prompt_usd_per_1k: float
    completion_usd_per_1k: float

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.prompt_usd_per_1k + (
            completion_tokens / 1000.0
        ) * self.completion_usd_per_1k


class LLMUsageTracker:
    """Tracks all LLM calls for cost/latency monitoring.

    Responsibilities:
    - Record each call (in-memory ring buffer)
    - Maintain rolling totals for cost
    - Emit Prometheus metrics (if available)
    - Invoke cost alert callbacks when thresholds are crossed
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        alert_budget_usd_per_window: Optional[float] = None,
        alert_window_s: Optional[int] = None,
        ring_buffer_size: int = 5000,
        prices: Optional[Dict[str, LLMPrices]] = None,
        log_path: Optional[str] = None,
    ):
        self.enabled = (
            bool(os.environ.get("LLM_USAGE_TRACKING_ENABLED", "1"))
            if enabled is None
            else enabled
        )
        self.alert_budget_usd_per_window = (
            float(os.environ.get("LLM_COST_ALERT_BUDGET_USD", "0"))
            if alert_budget_usd_per_window is None
            else alert_budget_usd_per_window
        )
        self.alert_window_s = int(
            os.environ.get("LLM_COST_ALERT_WINDOW_S", "3600")
            if alert_window_s is None
            else alert_window_s
        )
        self.ring_buffer_size = int(ring_buffer_size)
        self.prices = prices or {}

        self._lock = threading.Lock()
        self._events: List[dict] = []
        self._events_start_idx = 0

        self._window_start_ts = time.time()
        self._window_cost_usd = 0.0

        self._alert_callbacks: List[Callable[[dict], None]] = []

        self._prom = {}
        self._init_prometheus()

        self._log_path = log_path or os.environ.get(
            "LLM_USAGE_LOG_PATH", "./llm_usage_events.jsonl"
        )

    def _init_prometheus(self) -> None:
        if Counter is None:
            return

        self._prom["llm_calls_total"] = Counter(
            "astroml_llm_calls_total",
            "Total number of LLM calls",
            ["provider", "model"],
        )
        self._prom["llm_tokens_total"] = Counter(
            "astroml_llm_tokens_total",
            "Total tokens used by LLM calls",
            ["provider", "model", "token_type"],
        )
        self._prom["llm_latency_seconds"] = Histogram(
            "astroml_llm_latency_seconds",
            "Latency of LLM calls in seconds",
            ["provider", "model"],
        )
        self._prom["llm_cost_usd_total"] = Counter(
            "astroml_llm_cost_usd_total",
            "Cumulative cost in USD for LLM calls",
            ["provider", "model"],
        )
        self._prom["llm_cost_budget_usd_gauge"] = Gauge(
            "astroml_llm_cost_budget_usd_gauge",
            "Configured LLM cost budget per alert window (USD)",
        )
        try:
            if self.alert_budget_usd_per_window:
                self._prom["llm_cost_budget_usd_gauge"].set(
                    float(self.alert_budget_usd_per_window)
                )
        except Exception:
            pass

    def register_cost_alert_callback(self, cb: Callable[[dict], None]) -> None:
        """Register a callback invoked when budget is exceeded."""
        with self._lock:
            self._alert_callbacks.append(cb)

    def _push_event(self, event: dict) -> None:
        if len(self._events) < self.ring_buffer_size:
            self._events.append(event)
        else:
            # ring buffer: drop oldest
            self._events[self._events_start_idx % self.ring_buffer_size] = event
            self._events_start_idx += 1

    def _get_cost_from_prices_or_pass_through(self, usage: LLMUsage) -> float:
        # Cost_usd is preferred from provider.
        if usage.cost_usd is not None:
            return float(usage.cost_usd)

        key = f"{usage.provider}:{usage.model}"
        prices = self.prices.get(key) or self.prices.get(usage.model)
        if not prices:
            raise ValueError(
                "cost_usd missing and no prices configured for provider/model"
            )
        return prices.estimate_cost_usd(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    def record_call(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        cost_usd: Optional[float] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> LLMUsage:
        """Record an LLM call.

        All LLM calls should pass token counts and latency.
        If ``cost_usd`` is not provided, you must configure ``prices``.
        """

        total_tokens = int(prompt_tokens) + int(completion_tokens)

        usage = LLMUsage(
            provider=provider,
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total_tokens,
            cost_usd=float(cost_usd) if cost_usd is not None else None,  # type: ignore[arg-type]
            latency_s=float(latency_s),
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
        )

        # Resolve cost if needed
        resolved_cost_usd = (
            float(cost_usd)
            if cost_usd is not None
            else self._get_cost_from_prices_or_pass_through(usage)
        )

        usage_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_s": usage.latency_s,
            "cost_usd": resolved_cost_usd,
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
        }

        with self._lock:
            if not self.enabled:
                return usage

            self._push_event(usage_dict)
            self._window_cost_usd += resolved_cost_usd

            now = time.time()
            if now - self._window_start_ts >= self.alert_window_s:
                self._window_start_ts = now
                self._window_cost_usd = 0.0

            if self.alert_budget_usd_per_window and resolved_cost_usd is not None:
                # Trigger on window cost exceed
                if self._window_cost_usd >= float(self.alert_budget_usd_per_window):
                    alert = {
                        "type": "llm_cost_budget_exceeded",
                        "timestamp": usage_dict["timestamp"],
                        "budget_usd": float(self.alert_budget_usd_per_window),
                        "window_s": int(self.alert_window_s),
                        "window_cost_usd": float(self._window_cost_usd),
                        "last_call": usage_dict,
                    }
                    for cb in list(self._alert_callbacks):
                        try:
                            cb(alert)
                        except Exception as exc:  # pragma: no cover
                            logger.warning("LLM cost alert callback failed: %s", exc)

            # Emit Prometheus metrics
            prom = self._prom
            if prom:
                try:
                    prom["llm_calls_total"].labels(provider=provider, model=model).inc()
                    prom["llm_tokens_total"].labels(
                        provider=provider, model=model, token_type="prompt"
                    ).inc(usage.prompt_tokens)
                    prom["llm_tokens_total"].labels(
                        provider=provider, model=model, token_type="completion"
                    ).inc(usage.completion_tokens)
                    prom["llm_latency_seconds"].labels(
                        provider=provider, model=model
                    ).observe(usage.latency_s)
                    prom["llm_cost_usd_total"].labels(
                        provider=provider, model=model
                    ).inc(resolved_cost_usd)
                except Exception:
                    pass

        # Append JSONL log (all calls logged)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(usage_dict) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write LLM usage log: %s", exc)

        return LLMUsage(
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=resolved_cost_usd,
            latency_s=usage.latency_s,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
        )

    def recent_calls(self, limit: int = 100) -> List[dict]:
        """Return most recent recorded LLM call events."""
        with self._lock:
            if limit <= 0:
                return []
            return list(self._events[-limit:])


# Default process-wide tracker instance
default_llm_usage_tracker = LLMUsageTracker()

