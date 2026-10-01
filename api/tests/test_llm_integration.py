"""End-to-end integration tests for the LLM API gateway (#457).

Exercises every public LLM endpoint through the FastAPI ``TestClient``:
generate, streaming, chat, embeddings, RAG, model listing, cost usage,
health, feedback, and error handling. The gateway is wired to the in-process
mock provider, so no external provider calls are made.
"""

from __future__ import annotations

import json

from api.tests.llm_mocking import DeterministicLLMMock


def test_generate_returns_completion(client):
    response = client.post(
        "/api/v1/llm/generate",
        json={"prompt": "Summarise this transaction.", "model": "gpt-4-turbo"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["id"].startswith("gen_")
    assert payload["model"] == "gpt-4-turbo"
    assert isinstance(payload["content"], str) and payload["content"]

    usage = payload["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert payload["cost"] >= 0.0
    assert payload["latency_ms"] >= 0.0


def test_generate_rejects_missing_prompt(client):
    response = client.post("/api/v1/llm/generate", json={})
    assert response.status_code == 422


def test_generate_stream_emits_sse_chunks(client):
    with client.stream(
        "POST", "/api/v1/llm/generate/stream", json={"prompt": "Stream me please"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "data: " in body
    assert "data: [DONE]" in body

    deltas = [
        json.loads(line[len("data: ") :])["delta"]
        for line in body.splitlines()
        if line.startswith("data: ") and line[len("data: ") :] != "[DONE]"
    ]
    assert deltas
    assert all(isinstance(delta, str) for delta in deltas)


def test_chat_returns_assistant_message(client):
    response = client.post(
        "/api/v1/llm/chat",
        json={"messages": [{"role": "user", "content": "What is AstroML?"}]},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["message"]["role"] == "assistant"
    assert isinstance(payload["message"]["content"], str)
    assert payload["usage"]["total_tokens"] > 0


def test_models_list(client):
    response = client.get("/api/v1/llm/models")
    assert response.status_code == 200

    payload = response.json()
    assert isinstance(payload["models"], list) and payload["models"]
    assert payload["total"] == len(payload["models"])
    assert {"gpt-4-turbo", "gpt-4o"} <= {model["id"] for model in payload["models"]}


def test_embed_returns_vectors(client):
    response = client.post("/api/v1/llm/embed", json={"input": "Account activity"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["model"] == "text-embedding-3-small"
    assert isinstance(payload["embeddings"], list) and payload["embeddings"]
    assert len(payload["embeddings"][0]) == 1536


def test_rag_query_returns_documents(client):
    response = client.post(
        "/api/v1/llm/rag/query", json={"query": "Explain account risk", "top_k": 3}
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["query"] == "Explain account risk"
    assert isinstance(payload["answer"], str) and payload["answer"]
    assert len(payload["documents"]) == 3
    for document in payload["documents"]:
        assert {"doc_id", "content", "score", "metadata"} <= set(document)


def test_ask_returns_citations(client):
    response = client.post(
        "/api/v1/llm/ask",
        json={"question": "Where can I find the API usage examples and reference docs?"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["mode"] == "mock-rag"
    assert isinstance(payload["answer"], str) and payload["answer"]
    assert payload["citations"]
    assert {"source_id", "title", "url", "snippet"} <= set(payload["citations"][0])


def test_invalid_payload_rejected(client):
    assert client.post("/api/v1/llm/generate", json={}).status_code == 422
    assert client.post("/api/v1/llm/chat", json={"messages": []}).status_code == 422


def test_health_reports_all_providers(client):
    response = client.get("/api/v1/llm/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["overall_status"] in {"healthy", "degraded"}
    assert {"openai", "anthropic", "huggingface"} <= set(payload["providers"])
    assert "checked_at" in payload


def test_cost_usage_endpoint(client):
    client.post("/api/v1/llm/generate", json={"prompt": "account for this request"})

    response = client.get("/api/v1/llm/cost/usage")
    assert response.status_code == 200

    payload = response.json()
    assert payload["period"] == "all-time"
    assert payload["total_requests"] >= 0
    assert payload["total_tokens"] >= 0
    assert payload["total_cost_usd"] >= 0.0
    assert isinstance(payload["cost_by_model"], dict)


def test_feedback_round_trip(client):
    created = client.post(
        "/api/v1/llm/feedback",
        json={
            "feature": "generate",
            "prompt": "Explain account risk",
            "output": "An answer",
            "rating": 4,
        },
    )
    assert created.status_code == 201

    dashboard = client.get("/api/v1/llm/feedback/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["total"] >= 1


def test_mock_llm_metrics_and_latency():
    mock = DeterministicLLMMock(default_response="short answer", delay_ms=5)
    for i in range(5):
        assert mock.generate(f"prompt {i}")
    assert mock.p95_latency_ms() < 100
    assert mock.total_cost_usd() > 0
