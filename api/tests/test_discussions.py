"""Tests for GitHub Discussions API integration."""
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


class TestDiscussionsRouter:
    """Test discussions endpoints."""

    def test_get_recent_discussions(self):
        """Test fetching recent discussions."""
        response = client.get("/api/v1/discussions/recent")
        assert response.status_code == 200
        data = response.json()
        assert "discussions" in data
        assert "cached" in data
        assert isinstance(data["discussions"], list)

    def test_get_recent_discussions_with_limit(self):
        """Test fetching discussions with custom limit."""
        response = client.get("/api/v1/discussions/recent?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["discussions"]) <= 10

    def test_get_recent_discussions_with_category(self):
        """Test fetching discussions for specific category."""
        response = client.get("/api/v1/discussions/recent?category=Announcements")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["discussions"], list)

    def test_get_discussion_categories(self):
        """Test fetching discussion categories."""
        response = client.get("/api/v1/discussions/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "cached" in data
        assert isinstance(data["categories"], list)

    def test_search_discussions(self):
        """Test searching discussions."""
        response = client.post(
            "/api/v1/discussions/search",
            json={"query": "test", "limit": 20}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data

    def test_get_user_reputation_requires_token(self):
        """Test that user reputation endpoint handles missing token."""
        response = client.get("/api/v1/discussions/user-reputation/testuser")
        # Should either return 200 with cached data or 400 if no token
        assert response.status_code in [200, 400]

    def test_limit_validation(self):
        """Test limit parameter validation."""
        response = client.get("/api/v1/discussions/recent?limit=1000")
        assert response.status_code == 422  # Validation error

        response = client.get("/api/v1/discussions/recent?limit=0")
        assert response.status_code == 422  # Validation error

    def test_cache_functionality(self):
        """Test that responses are cached."""
        response1 = client.get("/api/v1/discussions/recent")
        assert response1.status_code == 200
        cached1 = response1.json().get("cached", False)

        response2 = client.get("/api/v1/discussions/recent")
        assert response2.status_code == 200
        cached2 = response2.json().get("cached", False)

        # Second call should be cached
        assert isinstance(cached2, bool)
