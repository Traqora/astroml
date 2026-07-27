"""Tests for the feedback collection endpoints — issue #308."""
from __future__ import annotations

import pytest

VALID = {"category": "bug", "message": "The chart fails to load on Safari."}


@pytest.mark.xdist_group("api_feedback")
class TestFeedback:
    def test_submit_creates_feedback(self, client):
        resp = client.post("/api/v1/feedback", json=VALID)
        assert resp.status_code == 201
        data = resp.json()
        assert data["category"] == "bug"
        assert data["status"] == "open"
        assert data["id"] > 0

    def test_submit_invalid_category_returns_422(self, client):
        resp = client.post("/api/v1/feedback", json={**VALID, "category": "spam"})
        assert resp.status_code == 422

    def test_submit_blank_message_returns_422(self, client):
        resp = client.post("/api/v1/feedback", json={**VALID, "message": "   "})
        assert resp.status_code == 422

    def test_submit_rejects_non_image_screenshot(self, client):
        resp = client.post(
            "/api/v1/feedback", json={**VALID, "screenshot": "not-a-data-url"}
        )
        assert resp.status_code == 422

    def test_submit_accepts_image_data_url(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={**VALID, "screenshot": "data:image/png;base64,AAAA"},
        )
        assert resp.status_code == 201

    def test_list_and_filter_by_category(self, client):
        client.post("/api/v1/feedback", json={"category": "bug", "message": "bug one"})
        client.post("/api/v1/feedback", json={"category": "feature", "message": "feat one"})

        resp = client.get("/api/v1/feedback?category=feature")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["category"] == "feature"

    def test_update_status_and_roadmap(self, client):
        fid = client.post(
            "/api/v1/feedback", json={"category": "feature", "message": "dark mode"}
        ).json()["id"]

        # Not on the roadmap while status is "open".
        assert client.get("/api/v1/feedback/roadmap").json()["planned"] == []

        upd = client.patch(f"/api/v1/feedback/{fid}", json={"status": "planned"})
        assert upd.status_code == 200
        assert upd.json()["status"] == "planned"

        roadmap = client.get("/api/v1/feedback/roadmap").json()
        assert any(item["id"] == fid for item in roadmap["planned"])

    def test_update_status_invalid_returns_422(self, client):
        fid = client.post("/api/v1/feedback", json=VALID).json()["id"]
        assert client.patch(f"/api/v1/feedback/{fid}", json={"status": "nope"}).status_code == 422

    def test_update_unknown_feedback_returns_404(self, client):
        assert client.patch("/api/v1/feedback/999999", json={"status": "planned"}).status_code == 404
