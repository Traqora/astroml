"""Fixtures for E2E tests."""
import pytest
import os
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def api_base_url():
    """Get API base URL."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def auth_token():
    """Get authentication token for tests."""
    return os.getenv("TEST_AUTH_TOKEN", None)


@pytest.fixture
def authenticated_client(client, auth_token):
    """Create authenticated test client."""
    if auth_token:
        client.headers.update({"Authorization": f"Bearer {auth_token}"})
    return client
