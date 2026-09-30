"""Tests for API security hardening."""

from __future__ import annotations

import os
import time

import pytest

from api.auth.hardening import (
    BruteForceProtector,
    IPRateLimiter,
    PublicRateLimitMiddleware,
    SecurityHeadersMiddleware,
    compute_token_fingerprint,
    enforce_jwt_secret,
    verify_token_fingerprint,
)


class TestBruteForceProtector:
    def test_not_locked_initially(self):
        protector = BruteForceProtector(max_failed_attempts=3, lockout_seconds=60)
        locked, retry_after = protector.is_locked("user1")
        assert locked is False
        assert retry_after is None

    def test_locks_after_max_attempts(self):
        protector = BruteForceProtector(max_failed_attempts=3, lockout_seconds=60)
        protector.record_failure("user1")
        protector.record_failure("user1")
        protector.record_failure("user1")

        locked, retry_after = protector.is_locked("user1")
        assert locked is True
        assert retry_after is not None
        assert retry_after > 0

    def test_does_not_lock_before_max(self):
        protector = BruteForceProtector(max_failed_attempts=5, lockout_seconds=60)
        for _ in range(4):
            protector.record_failure("user1")

        locked, _ = protector.is_locked("user1")
        assert locked is False

    def test_success_clears_failures(self):
        protector = BruteForceProtector(max_failed_attempts=3, lockout_seconds=60)
        protector.record_failure("user1")
        protector.record_failure("user1")
        protector.record_success("user1")

        locked, _ = protector.is_locked("user1")
        assert locked is False

    def test_lockout_expires(self):
        protector = BruteForceProtector(max_failed_attempts=2, lockout_seconds=0)
        protector.record_failure("user1")
        protector.record_failure("user1")
        time.sleep(0.01)

        locked, _ = protector.is_locked("user1")
        assert locked is False

    def test_separate_identifiers(self):
        protector = BruteForceProtector(max_failed_attempts=2, lockout_seconds=60)
        protector.record_failure("user1")
        protector.record_failure("user1")

        locked1, _ = protector.is_locked("user1")
        locked2, _ = protector.is_locked("user2")
        assert locked1 is True
        assert locked2 is False

    def test_cleanup_removes_expired(self):
        protector = BruteForceProtector(max_failed_attempts=2, lockout_seconds=0)
        protector.record_failure("user1")
        protector.record_failure("user1")
        time.sleep(0.01)

        removed = protector.cleanup()
        assert removed == 1

    def test_window_resets_count(self):
        protector = BruteForceProtector(max_failed_attempts=3, window_seconds=0, lockout_seconds=60)
        protector.record_failure("user1")
        protector.record_failure("user1")
        time.sleep(0.01)

        locked, _ = protector.is_locked("user1")
        assert locked is False


class TestIPRateLimiter:
    def test_allows_within_limit(self):
        limiter = IPRateLimiter(requests_per_minute=10)
        for _ in range(10):
            allowed, remaining, _ = limiter.is_allowed("192.168.1.1")
            assert allowed is True

    def test_blocks_over_limit(self):
        limiter = IPRateLimiter(requests_per_minute=5)
        for _ in range(5):
            limiter.is_allowed("192.168.1.1")

        allowed, remaining, retry_after = limiter.is_allowed("192.168.1.1")
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    def test_separate_ips(self):
        limiter = IPRateLimiter(requests_per_minute=2)
        limiter.is_allowed("192.168.1.1")
        limiter.is_allowed("192.168.1.1")

        allowed, _, _ = limiter.is_allowed("10.0.0.1")
        assert allowed is True

    def test_cleanup(self):
        limiter = IPRateLimiter(requests_per_minute=60)
        limiter.is_allowed("192.168.1.1")
        removed = limiter.cleanup()
        assert removed == 0


class TestTokenFingerprint:
    def test_fingerprint_deterministic(self):
        fp1 = compute_token_fingerprint("test-token-123")
        fp2 = compute_token_fingerprint("test-token-123")
        assert fp1 == fp2

    def test_different_tokens_different_fingerprints(self):
        fp1 = compute_token_fingerprint("token-a")
        fp2 = compute_token_fingerprint("token-b")
        assert fp1 != fp2

    def test_verify_correct_fingerprint(self):
        token = "my-secret-token"
        fp = compute_token_fingerprint(token)
        assert verify_token_fingerprint(token, fp) is True

    def test_verify_incorrect_fingerprint(self):
        assert verify_token_fingerprint("token-a", "wrong-fingerprint") is False


class TestEnforceJwtSecret:
    def test_dev_mode_allows_default(self, monkeypatch):
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        secret = enforce_jwt_secret()
        assert secret is not None

    def test_prod_requires_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="JWT_SECRET_KEY must be set"):
            enforce_jwt_secret()

    def test_prod_rejects_weak_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("JWT_SECRET_KEY", "change-me-in-production")
        with pytest.raises(RuntimeError, match="weak/default value"):
            enforce_jwt_secret()

    def test_prod_rejects_short_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("JWT_SECRET_KEY", "short")
        with pytest.raises(RuntimeError, match="at least 32 characters"):
            enforce_jwt_secret()

    def test_prod_accepts_strong_secret(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("JWT_SECRET_KEY", "a" * 32)
        secret = enforce_jwt_secret()
        assert secret == "a" * 32


class TestSecurityHeadersMiddleware:
    @pytest.mark.asyncio
    async def test_adds_security_headers(self):
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse

        middleware = SecurityHeadersMiddleware(None)

        async def call_next(request: Request):
            return PlainTextResponse("ok")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
        }
        request = Request(scope)
        response = await middleware.dispatch(request, call_next)

        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Pragma"] == "no-cache"


class TestPublicRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_allows_authenticated_requests(self):
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse

        middleware = PublicRateLimitMiddleware(None, requests_per_minute=1, burst_size=1)

        async def call_next(request: Request):
            return PlainTextResponse("ok")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/data",
            "headers": [(b"authorization", b"Bearer some-token")],
            "query_string": b"",
            "client": ("127.0.0.1", 8000),
        }
        request = Request(scope)
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limits_unauthenticated(self):
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse

        middleware = PublicRateLimitMiddleware(None, requests_per_minute=2, burst_size=2)

        async def call_next(request: Request):
            return PlainTextResponse("ok")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/data",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 8000),
        }
        request = Request(scope)

        response1 = await middleware.dispatch(request, call_next)
        response2 = await middleware.dispatch(request, call_next)
        response3 = await middleware.dispatch(request, call_next)

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 429
        assert "Retry-After" in response3.headers

    @pytest.mark.asyncio
    async def test_exempts_health_endpoints(self):
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse

        middleware = PublicRateLimitMiddleware(None, requests_per_minute=1, burst_size=1)

        async def call_next(request: Request):
            return PlainTextResponse("ok")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 8000),
        }
        request = Request(scope)

        for _ in range(10):
            response = await middleware.dispatch(request, call_next)
            assert response.status_code == 200


class TestMiddlewareStackBuilds:
    def test_starlette_can_construct_and_run_both(self):
        """Regression: Starlette builds the stack lazily, on the first request,
        as ``cls(app, **options)``. A middleware it cannot construct therefore
        surfaces as a 500 on every route rather than an import or startup error,
        so only an end-to-end request catches it. Mirrors the ``add_middleware``
        calls in ``api/app.py``.
        """
        from fastapi.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        app = Starlette(routes=[Route("/probe", lambda r: PlainTextResponse("ok"))])
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(PublicRateLimitMiddleware, requests_per_minute=30, burst_size=10)

        response = TestClient(app).get("/probe")

        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
