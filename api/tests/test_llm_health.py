"""Integration tests for LLM health endpoints."""
from __future__ import annotations


class TestLLMHealth:
    def test_llm_health_returns_200(self, client):
        resp = client.get("/api/v1/llm/health")
        assert resp.status_code == 200

    def test_llm_health_has_overall_status(self, client):
        data = client.get("/api/v1/llm/health").json()
        assert "overall_status" in data
        assert "providers" in data
        assert "checked_at" in data

    def test_llm_provider_health_endpoint(self, client):
        resp = client.get("/api/v1/llm/health/openai")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "openai"
        assert "status" in data
        assert "latency_ms" in data

    def test_llm_health_providers_include_expected(self, client):
        data = client.get("/api/v1/llm/health").json()
        assert "openai" in data["providers"]
        assert "anthropic" in data["providers"]
        assert "huggingface" in data["providers"]

    def test_prometheus_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "astroml_llm_provider_health" in resp.text
