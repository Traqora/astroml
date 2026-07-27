"""Integration tests for the LLM router."""
from __future__ import annotations


def test_llm_ask_returns_mock_rag_answer(client):
    response = client.post(
        "/api/v1/llm/ask",
        json={"question": "Where can I find the API usage examples and reference docs?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["mode"] == "mock-rag"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 20
    assert isinstance(data["citations"], list)
    assert data["citations"]

    citation_ids = {citation["source_id"] for citation in data["citations"]}
    assert "docs/api/usage-examples.md" in citation_ids
    assert "docs/api/reference.md" in citation_ids


def test_llm_ask_includes_router_context(client):
    response = client.post(
        "/api/v1/llm/ask",
        json={"question": "How do the LLM endpoints work in this API?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert any(
        citation["source_id"] == "api/routers/llm.py" for citation in data["citations"]
    )
