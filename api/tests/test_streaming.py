"""Tests for streaming API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.mark.xdist_group("api_streaming")
class TestStreamingEndpoints:
    def test_streaming_health_empty(self, client):
        """Test streaming health endpoint when no streams are registered."""
        resp = client.get("/api/v1/streaming/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_status"] == "degraded"
        assert data["last_updated"] is None
        assert len(data["streams"]) == 0

    def test_streaming_health_with_state(self):
        """Test streaming health endpoint with registered streams."""
        # Import here to avoid circular imports
        from api.routers.streaming import update_stream_state, _stream_state

        # Reset state first
        _stream_state.streams.clear()
        _stream_state.last_updated = None

        # Add a test stream
        update_stream_state("test-stream", {
            "stream_type": "effects",
            "horizon_url": "https://horizon-testnet.stellar.org",
            "is_healthy": True,
            "status": "active",
            "cursor": "123456789",
            "processed_count": 100,
            "consecutive_failures": 0,
            "current_backoff": 1.0,
            "lag_seconds": 5.0
        })

        # Test the endpoint
        with TestClient(app) as test_client:
            resp = test_client.get("/api/v1/streaming/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["overall_status"] == "healthy"
            assert data["last_updated"] is not None
            assert len(data["streams"]) == 1

            stream = data["streams"][0]
            assert stream["stream_id"] == "test-stream"
            assert stream["stream_type"] == "effects"
            assert stream["horizon_url"] == "https://horizon-testnet.stellar.org"
            assert stream["is_healthy"] is True
            assert stream["status"] == "active"
            assert stream["cursor"] == "123456789"
            assert stream["processed_count"] == 100
            assert stream["consecutive_failures"] == 0
            assert stream["current_backoff_seconds"] == 1.0
            assert stream["lag_seconds"] == 5.0
