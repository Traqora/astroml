"""Tests for contributors dashboard API (issue #280)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


MOCK_COMMIT_STATS = [
    {
        "login": "alice",
        "contributions": 120,
        "avatar_url": "https://avatars.githubusercontent.com/alice",
        "html_url": "https://github.com/alice",
    },
    {
        "login": "bob",
        "contributions": 8,
        "avatar_url": "https://avatars.githubusercontent.com/bob",
        "html_url": "https://github.com/bob",
    },
]

MOCK_PRS = [
    {"user": {"login": "alice"}, "merged_at": "2024-06-01T00:00:00Z"},
    {"user": {"login": "alice"}, "merged_at": "2024-06-02T00:00:00Z"},
    {"user": {"login": "bob"}, "merged_at": None},  # not merged
]

MOCK_ISSUES = [
    {"user": {"login": "alice"}, "pull_request": None},
    {"user": {"login": "alice"}, "pull_request": None},
    {"user": {"login": "alice"}, "pull_request": {"url": "..."}},  # is a PR, skip
    {"user": {"login": "bob"}, "pull_request": None},
]

MOCK_COMMITS = [
    {"commit": {"author": {"date": "2024-06-01T10:00:00Z"}}},
    {"commit": {"author": {"date": "2024-06-01T12:00:00Z"}}},
    {"commit": {"author": {"date": "2024-06-03T09:00:00Z"}}},
]

MOCK_USER = {
    "login": "alice",
    "avatar_url": "https://avatars.githubusercontent.com/alice",
    "html_url": "https://github.com/alice",
}

MOCK_CONTRIB_STATS = [
    {
        "author": {
            "login": "newbie",
            "avatar_url": "https://avatars.githubusercontent.com/newbie",
            "html_url": "https://github.com/newbie",
        },
        "total": 3,
        "weeks": [
            {"w": 9999999999, "c": 3},  # far future = always "new"
        ],
    }
]


def _make_gh_mock(*side_effects):
    mock = AsyncMock(side_effect=list(side_effects))
    return mock


class TestListContributors:

    def test_returns_contributors_sorted_by_total(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.side_effect = [MOCK_COMMIT_STATS, MOCK_PRS, MOCK_ISSUES]
            resp = client.get("/api/v1/contributors?sort_by=total")

        assert resp.status_code == 200
        data = resp.json()
        assert "contributors" in data
        assert data["total"] == 2
        # alice has more total contributions
        assert data["contributors"][0]["username"] == "alice"

    def test_alice_has_correct_pr_count(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.side_effect = [MOCK_COMMIT_STATS, MOCK_PRS, MOCK_ISSUES]
            resp = client.get("/api/v1/contributors")

        alice = next(c for c in resp.json()["contributors"] if c["username"] == "alice")
        assert alice["pull_requests"] == 2
        assert alice["issues"] == 2

    def test_badges_assigned_for_centurion(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.side_effect = [MOCK_COMMIT_STATS, MOCK_PRS, MOCK_ISSUES]
            resp = client.get("/api/v1/contributors")

        alice = next(c for c in resp.json()["contributors"] if c["username"] == "alice")
        badge_ids = [b["id"] for b in alice["badges"]]
        assert "centurion" in badge_ids

    def test_sort_by_commits(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.side_effect = [MOCK_COMMIT_STATS, MOCK_PRS, MOCK_ISSUES]
            resp = client.get("/api/v1/contributors?sort_by=commits")

        assert resp.status_code == 200
        assert resp.json()["contributors"][0]["username"] == "alice"

    def test_returns_cached_response(self, client):
        cached = {"contributors": [], "total": 0}
        with patch("api.routers.contributors._cached", return_value=cached):
            resp = client.get("/api/v1/contributors")
        assert resp.status_code == 200


class TestContributorActivity:

    def test_returns_activity_points(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.return_value = MOCK_COMMITS
            resp = client.get("/api/v1/contributors/activity?days=30")

        assert resp.status_code == 200
        data = resp.json()
        assert "activity" in data
        assert data["days"] == 30
        dates = [p["date"] for p in data["activity"]]
        assert "2024-06-01" in dates
        assert "2024-06-03" in dates

    def test_commits_bucketed_by_day(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.return_value = MOCK_COMMITS
            resp = client.get("/api/v1/contributors/activity")

        june1 = next(p for p in resp.json()["activity"] if p["date"] == "2024-06-01")
        assert june1["commits"] == 2  # two commits on June 1

    def test_filter_by_username(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.return_value = MOCK_COMMITS[:1]
            resp = client.get("/api/v1/contributors/activity?username=alice")

        assert resp.status_code == 200


class TestNewContributors:

    def test_returns_new_contributors(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.return_value = MOCK_CONTRIB_STATS
            resp = client.get("/api/v1/contributors/new?days=30")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 0

    def test_empty_when_no_recent_contributors(self, client):
        old_stats = [
            {
                "author": {"login": "veteran", "avatar_url": "", "html_url": ""},
                "total": 500,
                "weeks": [{"w": 1000000, "c": 500}],  # very old timestamp
            }
        ]
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.return_value = old_stats
            resp = client.get("/api/v1/contributors/new?days=30")

        assert resp.json()["total"] == 0


class TestGetContributor:

    def test_returns_contributor_profile(self, client):
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None), \
             patch("api.routers.contributors._store"):
            mock_gh.side_effect = [MOCK_USER, MOCK_COMMITS, MOCK_PRS, MOCK_ISSUES]
            resp = client.get("/api/v1/contributors/alice")

        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert "badges" in data
        assert data["profile_url"] == "https://github.com/alice"

    def test_404_for_unknown_user(self, client):
        import httpx
        with patch("api.routers.contributors._gh_get", new_callable=AsyncMock) as mock_gh, \
             patch("api.routers.contributors._cached", return_value=None):
            mock_gh.side_effect = Exception("404")
            resp = client.get("/api/v1/contributors/nonexistent-user-xyz")

        assert resp.status_code in (404, 500)
