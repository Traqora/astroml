"""LLM integration regression, latency/cost, chaos, and feedback tests (#401/#402)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from api.tests.llm_mocking import DeterministicLLMMock


GOLDEN_CASES = json.loads(
    (Path(__file__).parent / "llm_golden" / "ask_cases.json").read_text()
)


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_llm_ask_golden_regressions(client, case):
    response = client.post("/api/v1/llm/ask", json={"question": case["question"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == case["expected_mode"]
    citation_ids = {citation["source_id"] for citation in payload["citations"]}
    assert set(case["expected_citations"]).issubset(citation_ids)


def test_llm_latency_and_mock_cost_gates(client):
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        response = client.post("/api/v1/llm/ask", json={"question": "LLM API docs?"})
        timings.append(time.perf_counter() - start)
        assert response.status_code == 200

    p95 = sorted(timings)[int(0.95 * (len(timings) - 1))]
    assert p95 < 5.0

    mock = DeterministicLLMMock(default_response="short deterministic answer")
    for idx in range(10):
        assert mock.generate(f"prompt {idx}")
    assert mock.p95_latency_ms() < 50
    assert mock.total_cost_usd() < 0.01


def test_llm_chaos_invalid_payloads_and_provider_failure(client):
    assert client.post("/api/v1/llm/ask", json={}).status_code == 422
    assert client.post("/api/v1/llm/query", json={"query": ""}).status_code in {200, 400, 422}

    mock = DeterministicLLMMock(fail_for={"explode"})
    with pytest.raises(RuntimeError, match="injected LLM failure"):
        mock.generate("please explode")
    assert mock.generate("recover") == "mock llm response"


def test_llm_feedback_collection_dashboard_and_prompt_improvements(client):
    low = client.post(
        "/api/v1/llm/feedback",
        json={
            "feature": "ask",
            "prompt": "Explain account risk",
            "output": "Too vague",
            "rating": 2,
            "comment": "Needs citations",
        },
    )
    assert low.status_code == 201

    expert = client.post(
        "/api/v1/llm/feedback",
        json={
            "feature": "ask",
            "prompt": "Explain account risk",
            "output": "Cited answer",
            "rating": 5,
            "is_expert": True,
            "expert_weight": 3,
        },
    )
    assert expert.status_code == 201

    dashboard = client.get("/api/v1/llm/feedback/dashboard")
    assert dashboard.status_code == 200
    trend = dashboard.json()["trends"][0]
    assert trend["feature"] == "ask"
    assert trend["count"] == 2
    assert trend["expert_count"] == 1
    assert trend["weighted_average_rating"] > trend["average_rating"]

    improvements = client.get("/api/v1/llm/feedback/prompt-improvements")
    assert improvements.status_code == 200
    assert improvements.json()[0]["feature"] == "ask"
