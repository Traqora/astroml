"""Cost-aware tests for LLM features."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from astroml.llm.tracker import global_tracker
from astroml.llm.metrics import (
    LLM_COST_USD_TOTAL,
    LLM_REQUEST_LATENCY_SECONDS,
    LLM_TOKENS_TOTAL,
)


class TestLLMCostAware:
    """Tests to ensure LLM usage stays within cost and latency budgets."""

    def test_cost_threshold_not_exceeded(self, client: TestClient):
        baseline_cost = global_tracker.total_cost
        baseline_prom_cost = float(
            LLM_COST_USD_TOTAL._metrics.get("_value", {}).get("value", 0.0)
        )

        response = client.post(
            "/api/v1/llm/ask",
            json={"question": "What is the cost of this request?"},
        )
        assert response.status_code == 200

        new_cost = global_tracker.total_cost
        new_prom_cost = float(
            LLM_COST_USD_TOTAL._metrics.get("_value", {}).get("value", 0.0)
        )
        delta = (new_cost - baseline_cost) + (new_prom_cost - baseline_prom_cost)
        assert delta < 0.50, f"Single LLM request cost ${delta:.4f} exceeded $0.50 budget"

    def test_latency_budget(self, client: TestClient):
        start = time.perf_counter()
        response = client.post(
            "/api/v1/llm/ask",
            json={"question": "How fast is this response?"},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert response.status_code == 200
        assert elapsed_ms < 2000.0, f"LLM request took {elapsed_ms:.1f}ms, exceeded 2000ms budget"

    def test_token_budget(self, client: TestClient):
        baseline_tokens = global_tracker.total_prompt_tokens + global_tracker.total_completion_tokens
        baseline_prom_tokens = sum(
            v for v in LLM_TOKENS_TOTAL._metrics.get("_value", {}).values()
            if isinstance(v, (int, float))
        )

        response = client.post(
            "/api/v1/llm/ask",
            json={"question": "Count the tokens in this short question."},
        )
        assert response.status_code == 200

        new_tokens = global_tracker.total_prompt_tokens + global_tracker.total_completion_tokens
        delta = (new_tokens - baseline_tokens)
        assert delta <= 2000, f"Request used {delta} tokens, exceeded budget of 2000"

    def test_health_check_latency(self, client: TestClient):
        start = time.perf_counter()
        response = client.get("/api/v1/llm/health")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert response.status_code == 200
        assert elapsed_ms < 5000.0, f"Health check took {elapsed_ms:.1f}ms"
