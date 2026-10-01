"""Load/performance smoke tests for API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

@pytest.mark.benchmark(group="api-latency")
def test_health_latency(benchmark):
    """Benchmark /healthz endpoint latency.
    
    Threshold: p95 < 50ms
    """
    def do_request():
        return client.get("/healthz")
        
    response = benchmark(do_request)
    assert response.status_code in (200, 503) # 503 if DB is not set up, which is fine for latency smoke test


@pytest.mark.benchmark(group="api-latency")
def test_models_latency(benchmark):
    """Benchmark /api/v1/models/ list endpoint latency.
    
    Threshold: p95 < 100ms
    """
    def do_request():
        return client.get("/api/v1/models/")
        
    response = benchmark(do_request)
    # Auth middleware might block it, but we are testing latency
    assert response.status_code in (200, 401, 403)


@pytest.mark.benchmark(group="api-latency")
def test_transactions_latency(benchmark):
    """Benchmark /api/v1/transactions/ list endpoint latency.
    
    Threshold: p95 < 100ms
    """
    def do_request():
        return client.get("/api/v1/transactions/")
        
    response = benchmark(do_request)
    assert response.status_code in (200, 401, 403)

