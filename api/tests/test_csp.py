"""Tests for Content Security Policy (CSP) middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.csp import CSPMiddleware
from api.middleware.https import HSTSMiddleware, HTTPSRedirectMiddleware


@pytest.fixture
def app_with_csp() -> FastAPI:
    """Create a test FastAPI app with CSP middleware."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        CSPMiddleware,
        report_only=False,
        enable_nonce=True,
    )

    return app


@pytest.fixture
def app_with_csp_report_only() -> FastAPI:
    """Create a test FastAPI app with CSP report-only mode."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        CSPMiddleware,
        report_only=True,
        enable_nonce=True,
    )

    return app


@pytest.fixture
def app_without_nonce() -> FastAPI:
    """Create a test FastAPI app without nonce generation."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        CSPMiddleware,
        report_only=False,
        enable_nonce=False,
    )

    return app


def test_csp_header_present(app_with_csp: FastAPI) -> None:
    """Test that CSP header is added to responses."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    assert "Content-Security-Policy" in response.headers
    assert response.headers["Content-Security-Policy"] is not None


def test_csp_report_only_header(app_with_csp_report_only: FastAPI) -> None:
    """Test that CSP-Report-Only header is used in report-only mode."""
    client = TestClient(app_with_csp_report_only)
    response = client.get("/")

    assert "Content-Security-Policy-Report-Only" in response.headers
    assert "Content-Security-Policy" not in response.headers


def test_csp_default_src_self(app_with_csp: FastAPI) -> None:
    """Test that default-src is set to 'self'."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp


def test_csp_script_src_with_nonce(app_with_csp: FastAPI) -> None:
    """Test that script-src includes nonce when enabled."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "'nonce-" in csp


def test_csp_script_src_without_nonce(app_without_nonce: FastAPI) -> None:
    """Test that script-src does not include nonce when disabled."""
    client = TestClient(app_without_nonce)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "'nonce-" not in csp


def test_csp_style_src_unsafe_inline(app_with_csp: FastAPI) -> None:
    """Test that style-src allows unsafe-inline for development."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "style-src 'self' 'unsafe-inline'" in csp


def test_csp_img_src(app_with_csp: FastAPI) -> None:
    """Test that img-src allows self, data, and https."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "img-src 'self' data: https:" in csp


def test_csp_connect_src_self(app_with_csp: FastAPI) -> None:
    """Test that connect-src is restricted to self."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "connect-src 'self'" in csp


def test_csp_object_src_none(app_with_csp: FastAPI) -> None:
    """Test that object-src is set to none to block plugins."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "object-src 'none'" in csp


def test_csp_frame_ancestors_none(app_with_csp: FastAPI) -> None:
    """Test that frame-ancestors is set to none to prevent clickjacking."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp


def test_csp_nonce_header(app_with_csp: FastAPI) -> None:
    """Test that nonce is added to response headers."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    assert "X-CSP-Nonce" in response.headers
    assert response.headers["X-CSP-Nonce"] is not None
    assert len(response.headers["X-CSP-Nonce"]) > 0


def test_csp_additional_security_headers(app_with_csp: FastAPI) -> None:
    """Test that additional security headers are added."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Permissions-Policy" in response.headers


def test_csp_permissions_policy(app_with_csp: FastAPI) -> None:
    """Test that Permissions-Policy restricts sensitive features."""
    client = TestClient(app_with_csp)
    response = client.get("/")

    permissions = response.headers["Permissions-Policy"]
    assert "geolocation=()" in permissions
    assert "microphone=()" in permissions
    assert "camera=()" in permissions


def test_csp_nonce_different_per_request(app_with_csp: FastAPI) -> None:
    """Test that nonce is different for each request."""
    client = TestClient(app_with_csp)

    response1 = client.get("/")
    response2 = client.get("/")

    nonce1 = response1.headers["X-CSP-Nonce"]
    nonce2 = response2.headers["X-CSP-Nonce"]

    assert nonce1 != nonce2


def test_csp_hash_script() -> None:
    """Test script hash generation."""
    from api.middleware.csp import CSPMiddleware

    script = "console.log('test');"
    hash_value = CSPMiddleware.hash_script(script)

    assert hash_value is not None
    assert isinstance(hash_value, str)
    assert len(hash_value) > 0

    # Same script should produce same hash
    hash_value2 = CSPMiddleware.hash_script(script)
    assert hash_value == hash_value2

    # Different script should produce different hash
    different_script = "console.log('different');"
    different_hash = CSPMiddleware.hash_script(different_script)
    assert hash_value != different_hash


def test_csp_report_uri() -> None:
    """Test CSP with report URI configured."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    app.add_middleware(
        CSPMiddleware,
        report_only=False,
        enable_nonce=True,
        report_uri="https://example.com/csp-report",
    )

    client = TestClient(app)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]
    assert "report-uri https://example.com/csp-report" in csp


def test_csp_missing_on_https_redirect_response() -> None:
    """Regression test for a real gap in the production middleware stack.

    ``api/app.py`` registers middleware in this order: ``CSPMiddleware``,
    then ``HTTPSRedirectMiddleware``, then ``HSTSMiddleware``. Starlette
    wraps middleware so the *first* ``add_middleware`` call ends up
    *innermost* — meaning a plain-HTTP request is intercepted by
    ``HTTPSRedirectMiddleware`` (which issues a 301 and returns immediately,
    without calling ``call_next``) *before* it ever reaches
    ``CSPMiddleware``. The practical effect: the 301 redirect response that
    tells a browser to upgrade to HTTPS carries **no CSP header at all**.

    This test pins that current, verified behavior down as an explicit
    regression test (rather than leaving it as an undocumented gap) so a
    future change to this interaction — in either direction — is a
    deliberate decision instead of an accidental side effect. If this
    assertion ever needs to flip (e.g. because CSP is deliberately applied
    even to redirects for defense-in-depth), that's a conscious call for
    whoever touches this stack next, not a silent regression.
    """
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "test"}

    # Mirrors api/app.py's registration order exactly.
    app.add_middleware(CSPMiddleware, report_only=False, enable_nonce=True)
    app.add_middleware(HTTPSRedirectMiddleware, enabled=True, allowed_hosts=None)
    app.add_middleware(HSTSMiddleware, max_age=31536000, enabled=True)

    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 301
    assert "Content-Security-Policy" not in response.headers
    assert "Content-Security-Policy-Report-Only" not in response.headers

    # Once the client follows the redirect onto HTTPS, CSP is present again
    # — confirming the gap is specific to the redirect hop, not a general
    # failure of the middleware to ever fire.
    https_response = client.get("https://testserver/")
    assert "Content-Security-Policy" in https_response.headers
