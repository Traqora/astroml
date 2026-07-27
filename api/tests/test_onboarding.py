"""Tests for contributor onboarding checklist endpoints — Issue #281."""
from __future__ import annotations

import pytest


@pytest.mark.xdist_group("api_onboarding")
class TestOnboardingProgress:
    def test_get_new_contributor_returns_empty_checklist(self, client):
        resp = client.get("/api/v1/contributors/onboarding/test_user_xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["github_username"] == "test_user_xyz"
        assert data["completed_count"] == 0
        assert data["is_complete"] is False
        assert isinstance(data["checklist"], list)
        assert len(data["checklist"]) == 4

    def test_checklist_items_have_required_fields(self, client):
        data = client.get("/api/v1/contributors/onboarding/test_user_xyz").json()
        for item in data["checklist"]:
            assert "step" in item
            assert "label" in item
            assert "completed" in item
            assert item["completed"] is False

    def test_complete_step_marks_it_done(self, client):
        resp = client.post(
            "/api/v1/contributors/onboarding/test_user_abc/complete",
            json={"step": "fork_repo"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["completed_count"] == 1
        fork_item = next(i for i in data["checklist"] if i["step"] == "fork_repo")
        assert fork_item["completed"] is True

    def test_complete_unknown_step_returns_422(self, client):
        resp = client.post(
            "/api/v1/contributors/onboarding/test_user_abc/complete",
            json={"step": "nonexistent_step"},
        )
        assert resp.status_code == 422

    def test_progress_pct_increases_with_steps(self, client):
        username = "test_pct_user"
        for step in ["fork_repo", "setup_dev_environment", "run_tests", "first_pr"]:
            client.post(
                f"/api/v1/contributors/onboarding/{username}/complete",
                json={"step": step},
            )
        data = client.get(f"/api/v1/contributors/onboarding/{username}").json()
        assert data["progress_pct"] == 100
        assert data["is_complete"] is True

    def test_list_all_returns_list(self, client):
        resp = client.get("/api/v1/contributors/onboarding")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
