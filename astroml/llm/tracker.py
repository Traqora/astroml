"""LLM Token Usage and Cost Tracking."""
import logging
from typing import Dict

from astroml.llm.metrics import (
    LLM_COST_USD_TOTAL,
    LLM_REQUEST_LATENCY_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
)

logger = logging.getLogger(__name__)

COST_RATES = {
    "openai": {"prompt": 0.03, "completion": 0.06},
    "anthropic": {"prompt": 0.015, "completion": 0.075},
    "huggingface": {"prompt": 0.001, "completion": 0.001},
}


class LLMUsageTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.alert_threshold = 100.0  # $100

    def record_usage(
        self, provider_name: str, usage: Dict[str, int], latency_ms: float
    ) -> float:
        rates = COST_RATES.get(
            provider_name.lower(), {"prompt": 0.0, "completion": 0.0}
        )
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = (prompt_tokens / 1000.0) * rates["prompt"] + (
            completion_tokens / 1000.0
        ) * rates["completion"]

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost

        LLM_REQUESTS_TOTAL.labels(
            provider=provider_name, status="success"
        ).inc()
        LLM_REQUEST_LATENCY_SECONDS.labels(provider=provider_name).observe(
            latency_ms / 1000.0
        )
        LLM_COST_USD_TOTAL.labels(provider=provider_name).inc(cost)
        LLM_TOKENS_TOTAL.labels(
            provider=provider_name, token_type="prompt"
        ).inc(prompt_tokens)
        LLM_TOKENS_TOTAL.labels(
            provider=provider_name, token_type="completion"
        ).inc(completion_tokens)

        logger.info(
            "LLM Usage Recorded: Provider=%s, PromptTokens=%d, "
            "CompletionTokens=%d, Cost=$%.4f, Latency=%.2fms",
            provider_name,
            prompt_tokens,
            completion_tokens,
            cost,
            latency_ms,
        )

        self.check_alerts()
        return cost

    def record_error(self, provider_name: str) -> None:
        LLM_REQUESTS_TOTAL.labels(provider=provider_name, status="error").inc()

    def check_alerts(self):
        if self.total_cost > self.alert_threshold:
            logger.warning(
                "LLM Cost Alert! Total cost ($%.2f) has exceeded "
                "threshold ($%.2f)",
                self.total_cost,
                self.alert_threshold,
            )

    def get_summary(self) -> Dict[str, float]:
        return {
            "total_cost": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens
            + self.total_completion_tokens,
        }


global_tracker = LLMUsageTracker()
