"""
RFC 7807 error conformance tests (issue #955).

Covers: every exception handler (HTTPException, RequestValidationError,
unhandled) returns application/problem+json with the required RFC 7807
members (type, title, status), and correct instance/detail binding.

Uses a minimal standalone FastAPI app with the real handlers from
api.middleware.errors registered, rather than importing the full api.app.
The handlers themselves are what this test verifies, and they don't
depend on the database, GraphQL schema, or any other app-level wiring;
isolating them here also means this test doesn't depend on the api.app
import chain (which pulls in the DB layer, GraphQL schema, auth, etc.)
succeeding at all.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from api.middleware.errors import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)

RFC7807_CONTENT_TYPE = "application/problem+json"
REQUIRED_MEMBERS = {"type", "title", "status"}


class _ReportBody(BaseModel):
    message: str = Field(..., min_length=1)


def _build_test_app() -> FastAPI:
    """A minimal app wired with the same exception handlers api.app uses."""
    app = FastAPI()
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.post("/api/v1/errors/report")
    async def report(body: _ReportBody):
        return {"ok": True}

    @app.get("/api/v1/accounts/{public_key}")
    async def get_account(public_key: str):
        raise HTTPException(status_code=404, detail=f"Account {public_key!r} not found")

    @app.get("/api/v1/boom")
    async def boom():
        raise RuntimeError("unexpected failure")

    return app


@pytest.fixture()
def rfc7807_client():
    with TestClient(_build_test_app(), raise_server_exceptions=False) as c:
        yield c


@pytest.mark.xdist_group("api_errors")
class TestValidationErrorConformance:
    """422 responses from RequestValidationError."""

    def test_content_type_is_problem_json(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        assert resp.headers["content-type"].startswith(RFC7807_CONTENT_TYPE)

    def test_has_required_rfc7807_members(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        body = resp.json()
        assert REQUIRED_MEMBERS.issubset(body.keys())

    def test_status_matches_http_status_code(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        assert resp.json()["status"] == resp.status_code == 422

    def test_type_is_a_uri(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        problem_type = resp.json()["type"]
        assert problem_type.startswith("https://") or problem_type == "about:blank"

    def test_instance_is_the_request_path(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        assert resp.json()["instance"] == "/api/v1/errors/report"

    def test_details_carries_field_level_errors(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        body = resp.json()
        assert isinstance(body["details"], list)
        assert len(body["details"]) > 0


@pytest.mark.xdist_group("api_errors")
class TestHttpExceptionConformance:
    """4xx responses from an explicit HTTPException (e.g. 404)."""

    def test_not_found_is_problem_json(self, rfc7807_client):
        resp = rfc7807_client.get("/api/v1/accounts/DOES_NOT_EXIST")
        assert resp.headers["content-type"].startswith(RFC7807_CONTENT_TYPE)

    def test_not_found_has_required_members(self, rfc7807_client):
        resp = rfc7807_client.get("/api/v1/accounts/DOES_NOT_EXIST")
        body = resp.json()
        assert REQUIRED_MEMBERS.issubset(body.keys())
        assert body["status"] == resp.status_code == 404

    def test_not_found_instance_is_request_path(self, rfc7807_client):
        resp = rfc7807_client.get("/api/v1/accounts/DOES_NOT_EXIST")
        assert resp.json()["instance"] == "/api/v1/accounts/DOES_NOT_EXIST"


@pytest.mark.xdist_group("api_errors")
class TestUnhandledExceptionConformance:
    """500 responses from an unhandled exception."""

    def test_unhandled_is_problem_json(self, rfc7807_client):
        resp = rfc7807_client.get("/api/v1/boom")
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith(RFC7807_CONTENT_TYPE)

    def test_unhandled_has_required_members_and_no_leaked_detail(self, rfc7807_client):
        resp = rfc7807_client.get("/api/v1/boom")
        body = resp.json()
        assert REQUIRED_MEMBERS.issubset(body.keys())
        assert body["status"] == 500
        # The real exception message must never reach the client.
        assert "unexpected failure" not in resp.text


@pytest.mark.xdist_group("api_errors")
class TestProblemDetailBackwardCompatibility:
    """The prior {code, message, details} shape's information is preserved
    as RFC 7807 extension members, so existing consumers reading `code` or
    `details` are not broken by the format migration."""

    def test_code_extension_member_present(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        assert resp.json()["code"] == "VALIDATION_ERROR"

    def test_detail_carries_the_former_message(self, rfc7807_client):
        resp = rfc7807_client.post("/api/v1/errors/report", json={"message": ""})
        assert resp.json()["detail"] == "Request validation failed"
