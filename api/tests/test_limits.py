"""
Request size and pagination limit tests (issue #1247).

Covers:
  1. Request size — expected path: body under limit → 200.
  2. Request size — failure case: body over limit → 413 problem+json.
  3. Pagination — expected path: limit within max_page_size → 200.
  4. Pagination — failure case: limit above max_page_size → 400 problem+json.

Uses a minimal standalone FastAPI app with no DB, Redis, or external
services, following the same pattern as test_rfc7807_errors.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.middleware.errors import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from api.middleware.pagination import clamp_page_size
from api.middleware.size_limit import ContentSizeLimitMiddleware
from api.config import Settings

PROBLEM_CONTENT_TYPE = "application/problem+json"

# A test-scoped Settings object with a small max_page_size so we don't have
# to send 101 items; it also isolates from whatever the real settings say.
_TEST_SETTINGS = Settings(
    max_request_body_size=100,  # intentionally tiny for the test
    max_page_size=100,
    default_page_size=20,
)
_TEST_SIZE_LIMIT = 100  # bytes — matches _TEST_SETTINGS.max_request_body_size


def _build_test_app() -> FastAPI:
    """Minimal FastAPI app wired with size and pagination limits."""
    app = FastAPI()
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Wire the size-limit middleware with the test limit.
    app.add_middleware(ContentSizeLimitMiddleware, max_body_size=_TEST_SIZE_LIMIT)

    @app.post("/api/v1/echo")
    async def echo(request: Request):
        body = await request.body()
        return {"length": len(body)}

    @app.get("/api/v1/items")
    async def list_items(limit: int = Query(default=20, ge=1)):
        clamp_page_size(limit, _settings=_TEST_SETTINGS)
        return {"items": [], "limit": limit}

    return app


@pytest.fixture(scope="module")
def limits_client():
    with TestClient(_build_test_app(), raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Request size — expected path
# ---------------------------------------------------------------------------

class TestRequestSizeAllowed:
    def test_small_body_returns_200(self, limits_client):
        """A body well under the limit should pass through normally."""
        payload = b"x" * 50  # 50 bytes < 100-byte limit
        resp = limits_client.post(
            "/api/v1/echo",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200

    def test_small_body_response_has_correct_length(self, limits_client):
        payload = b"y" * 50
        resp = limits_client.post(
            "/api/v1/echo",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.json()["length"] == 50


# ---------------------------------------------------------------------------
# 2. Request size — failure case
# ---------------------------------------------------------------------------

class TestRequestSizeExceeded:
    def test_oversized_body_returns_413(self, limits_client):
        """A body exceeding the limit must be rejected with 413."""
        payload = b"z" * 200  # 200 bytes > 100-byte limit
        resp = limits_client.post(
            "/api/v1/echo",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 413

    def test_oversized_body_content_type_is_problem_json(self, limits_client):
        payload = b"z" * 200
        resp = limits_client.post(
            "/api/v1/echo",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)

    def test_oversized_body_code_is_request_too_large(self, limits_client):
        payload = b"z" * 200
        resp = limits_client.post(
            "/api/v1/echo",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        body = resp.json()
        assert body["code"] == "REQUEST_TOO_LARGE"

    def test_oversized_body_has_rfc7807_members(self, limits_client):
        payload = b"z" * 200
        resp = limits_client.post(
            "/api/v1/echo",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        body = resp.json()
        assert {"type", "title", "status", "code"}.issubset(body.keys())
        assert body["status"] == 413


# ---------------------------------------------------------------------------
# 3. Pagination — expected path
# ---------------------------------------------------------------------------

class TestPaginationAllowed:
    def test_limit_within_max_returns_200(self, limits_client):
        """limit=20 is within max_page_size=100 and should succeed."""
        resp = limits_client.get("/api/v1/items?limit=20")
        assert resp.status_code == 200

    def test_limit_at_max_returns_200(self, limits_client):
        """limit=100 equals max_page_size=100 and should succeed."""
        resp = limits_client.get("/api/v1/items?limit=100")
        assert resp.status_code == 200

    def test_limit_echoed_in_response(self, limits_client):
        resp = limits_client.get("/api/v1/items?limit=20")
        assert resp.json()["limit"] == 20


# ---------------------------------------------------------------------------
# 4. Pagination — failure case
# ---------------------------------------------------------------------------

class TestPaginationExceeded:
    def test_limit_above_max_returns_400(self, limits_client):
        """limit=200 exceeds max_page_size=100 and must return 400."""
        resp = limits_client.get("/api/v1/items?limit=200")
        assert resp.status_code == 400

    def test_limit_above_max_content_type_is_problem_json(self, limits_client):
        resp = limits_client.get("/api/v1/items?limit=200")
        assert resp.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)

    def test_limit_above_max_code_is_page_size_too_large(self, limits_client):
        resp = limits_client.get("/api/v1/items?limit=200")
        body = resp.json()
        assert body["code"] == "PAGE_SIZE_TOO_LARGE"

    def test_limit_above_max_has_rfc7807_members(self, limits_client):
        resp = limits_client.get("/api/v1/items?limit=200")
        body = resp.json()
        assert {"type", "title", "status", "code"}.issubset(body.keys())
        assert body["status"] == 400
