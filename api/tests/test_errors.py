"""
Integration tests for the frontend error logging endpoint — issue #292.

Covers: happy path, empty message rejection, missing optional fields,
oversized fields, and that the endpoint never returns 5xx.
"""
from __future__ import annotations

import pytest


@pytest.mark.xdist_group("api_errors")
class TestErrorReportEndpoint:
    """POST /api/v1/errors/report"""

    def test_valid_report_returns_204(self, client):
        resp = client.post(
            "/api/v1/errors/report",
            json={
                "message": "TypeError: Cannot read property 'x' of undefined",
                "stack": "at Component.render (App.tsx:42)\n  at ...",
                "boundary": "Loyalty Dashboard",
                "url": "http://localhost:5173/",
                "user_agent": "Mozilla/5.0 (test)",
                "timestamp": "2026-06-23T12:00:00Z",
            },
        )
        assert resp.status_code == 204

    def test_minimal_report_returns_204(self, client):
        """Only `message` is required — all other fields are optional."""
        resp = client.post(
            "/api/v1/errors/report",
            json={"message": "ChunkLoadError: Loading chunk 3 failed."},
        )
        assert resp.status_code == 204

    def test_empty_message_is_rejected(self, client):
        resp = client.post("/api/v1/errors/report", json={"message": "   "})
        assert resp.status_code == 422

    def test_missing_message_is_rejected(self, client):
        resp = client.post("/api/v1/errors/report", json={"boundary": "App"})
        assert resp.status_code == 422

    def test_oversized_message_is_rejected(self, client):
        resp = client.post(
            "/api/v1/errors/report",
            json={"message": "x" * 2001},
        )
        assert resp.status_code == 422

    def test_extra_metadata_accepted(self, client):
        resp = client.post(
            "/api/v1/errors/report",
            json={
                "message": "Error in chart renderer",
                "extra": {"dataPoints": 12000, "chartType": "LineChart"},
            },
        )
        assert resp.status_code == 204

    def test_component_stack_accepted(self, client):
        resp = client.post(
            "/api/v1/errors/report",
            json={
                "message": "Invariant violation",
                "component_stack": "  at FraudDetectionPanel\n  at ErrorBoundary\n  at App",
            },
        )
        assert resp.status_code == 204

    def test_endpoint_never_returns_500(self, client):
        """Malformed JSON body should return 422 (validation), never 500."""
        resp = client.post(
            "/api/v1/errors/report",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code != 500
