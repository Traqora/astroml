"""Tests for HTTPS enforcement middleware."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.middleware.https import HTTPSRedirectMiddleware, HSTSMiddleware


@pytest.fixture
def app_with_https_redirect() -> FastAPI:
    """Create a test FastAPI app with HTTPS redirect enabled."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    @app.get("/api/test")
    async def test_endpoint():
        return {"status": "ok"}

    app.add_middleware(
        HTTPSRedirectMiddleware,
        enabled=True,
        allowed_hosts=None,
    )

    return app


@pytest.fixture
def app_with_https_redirect_disabled() -> FastAPI:
    """Create a test FastAPI app with HTTPS redirect disabled."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        HTTPSRedirectMiddleware,
        enabled=False,
    )

    return app


@pytest.fixture
def app_with_https_redirect_allowed_hosts() -> FastAPI:
    """Create a test FastAPI app with HTTPS redirect and allowed hosts."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        HTTPSRedirectMiddleware,
        enabled=True,
        allowed_hosts=["example.com", "api.example.com"],
    )

    return app


@pytest.fixture
def app_with_hsts() -> FastAPI:
    """Create a test FastAPI app with HSTS enabled."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        HSTSMiddleware,
        max_age=31536000,
        include_subdomains=True,
        preload=False,
        enabled=True,
    )

    return app


@pytest.fixture
def app_with_hsts_disabled() -> FastAPI:
    """Create a test FastAPI app with HSTS disabled."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        HSTSMiddleware,
        enabled=False,
    )

    return app


def test_https_redirect_http_request(app_with_https_redirect: FastAPI) -> None:
    """Test that HTTP requests are redirected to HTTPS."""
    client = TestClient(app_with_https_redirect, base_url="http://testserver")
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 301
    assert "location" in response.headers
    assert response.headers["location"].startswith("https://")


def test_https_redirect_https_request(app_with_https_redirect: FastAPI) -> None:
    """Test that HTTPS requests are not redirected."""
    client = TestClient(app_with_https_redirect, base_url="https://testserver")
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "test"}


def test_https_redirect_disabled(app_with_https_redirect_disabled: FastAPI) -> None:
    """Test that HTTPS redirect can be disabled."""
    client = TestClient(app_with_https_redirect_disabled, base_url="http://testserver")
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "test"}


def test_https_redirect_allowed_hosts_match(app_with_https_redirect_allowed_hosts: FastAPI) -> None:
    """Test that HTTPS redirect works for allowed hosts."""
    client = TestClient(app_with_https_redirect_allowed_hosts, base_url="http://example.com")
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 301
    assert "location" in response.headers


def test_https_redirect_allowed_hosts_no_match(app_with_https_redirect_allowed_hosts: FastAPI) -> None:
    """Test that HTTPS redirect is skipped for non-allowed hosts."""
    client = TestClient(app_with_https_redirect_allowed_hosts, base_url="http://other.com")
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "test"}


def test_hsts_header_present_https(app_with_hsts: FastAPI) -> None:
    """Test that HSTS header is added for HTTPS requests."""
    client = TestClient(app_with_hsts, base_url="https://testserver")
    response = client.get("/")

    assert "Strict-Transport-Security" in response.headers
    assert response.headers["Strict-Transport-Security"] is not None


def test_hsts_header_not_present_http(app_with_hsts: FastAPI) -> None:
    """Test that HSTS header is not added for HTTP requests."""
    client = TestClient(app_with_hsts, base_url="http://testserver")
    response = client.get("/")

    assert "Strict-Transport-Security" not in response.headers


def test_hsts_header_disabled(app_with_hsts_disabled: FastAPI) -> None:
    """Test that HSTS header is not added when disabled."""
    client = TestClient(app_with_hsts_disabled, base_url="https://testserver")
    response = client.get("/")

    assert "Strict-Transport-Security" not in response.headers


def test_hsts_max_age(app_with_hsts: FastAPI) -> None:
    """Test that HSTS max-age is set correctly."""
    client = TestClient(app_with_hsts, base_url="https://testserver")
    response = client.get("/")

    hsts_header = response.headers["Strict-Transport-Security"]
    assert "max-age=31536000" in hsts_header


def test_hsts_include_subdomains(app_with_hsts: FastAPI) -> None:
    """Test that HSTS includes subdomains when configured."""
    client = TestClient(app_with_hsts, base_url="https://testserver")
    response = client.get("/")

    hsts_header = response.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in hsts_header


def test_hsts_preload(app_with_hsts: FastAPI) -> None:
    """Test that HSTS preload flag is set when configured."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        HSTSMiddleware,
        max_age=31536000,
        include_subdomains=True,
        preload=True,
        enabled=True,
    )

    client = TestClient(app, base_url="https://testserver")
    response = client.get("/")

    hsts_header = response.headers["Strict-Transport-Security"]
    assert "preload" in hsts_header


def test_hsts_custom_max_age() -> None:
    """Test that custom HSTS max-age is respected."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        HSTSMiddleware,
        max_age=86400,  # 1 day
        include_subdomains=False,
        preload=False,
        enabled=True,
    )

    client = TestClient(app, base_url="https://testserver")
    response = client.get("/")

    hsts_header = response.headers["Strict-Transport-Security"]
    assert "max-age=86400" in hsts_header
    assert "includeSubDomains" not in hsts_header
    assert "preload" not in hsts_header


def test_https_redirect_preserves_path(app_with_https_redirect: FastAPI) -> None:
    """Test that HTTPS redirect preserves the request path."""
    client = TestClient(app_with_https_redirect, base_url="http://testserver")
    response = client.get("/api/test", follow_redirects=False)

    assert response.status_code == 301
    assert response.headers["location"] == "https://testserver/api/test"


def test_https_redirect_preserves_query_params(app_with_https_redirect: FastAPI) -> None:
    """Test that HTTPS redirect preserves query parameters."""
    client = TestClient(app_with_https_redirect, base_url="http://testserver")
    response = client.get("/?param=value", follow_redirects=False)

    assert response.status_code == 301
    assert "param=value" in response.headers["location"]
    assert response.headers["location"].startswith("https://")
