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


class TestAuthMiddlewareLockout:
    """End-to-end lockout enforcement through AuthMiddleware.dispatch.

    TestBruteForceProtector above covers the counter in isolation; nothing
    exercised the middleware actually consulting it on the request path, so
    a wiring bug (wrong lock key, lockout check skipped, success not
    clearing it) could ship unnoticed.
    """

    @pytest.fixture(autouse=True)
    def _isolated_protector(self, monkeypatch):
        import api.auth.hardening as hardening_module
        import api.auth.middleware as middleware_module

        protector = BruteForceProtector(max_failed_attempts=3, lockout_seconds=900)
        monkeypatch.setattr(hardening_module, "_brute_force_protector", protector)
        monkeypatch.setattr(middleware_module, "_brute_force_protector", protector)
        return protector

    def _make_middleware(self, monkeypatch, *, auth_succeeds: bool):
        import api.auth.middleware as middleware_module

        monkeypatch.setattr(middleware_module, "is_auth_enabled", lambda: True)
        monkeypatch.setattr(middleware_module, "PUBLIC_PATHS", frozenset())

        def fake_authenticate_token(token, session):
            if auth_succeeds:

                class _Auth:
                    auth_type = "jwt"
                    subject = "user1"

                return _Auth()
            raise ValueError("bad token")

        monkeypatch.setattr(middleware_module, "authenticate_token", fake_authenticate_token)
        monkeypatch.setattr(
            middleware_module, "_sync_session_factory", lambda: (lambda: _NullSession())
        )
        return middleware_module.AuthMiddleware(None)

    def _request(self, *, client_ip: str = "10.0.0.1") -> "Request":
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/protected",
            "headers": [(b"authorization", b"Bearer sometoken123456")],
            "query_string": b"",
            "client": (client_ip, 8000),
        }
        return Request(scope)

    @pytest.mark.asyncio
    async def test_locks_out_after_repeated_failures(self, monkeypatch):
        from starlette.responses import PlainTextResponse

        middleware = self._make_middleware(monkeypatch, auth_succeeds=False)

        async def call_next(request):
            return PlainTextResponse("ok")

        responses = [await middleware.dispatch(self._request(), call_next) for _ in range(3)]
        assert [r.status_code for r in responses] == [401, 401, 401]

        locked_response = await middleware.dispatch(self._request(), call_next)
        assert locked_response.status_code == 429
        assert "Retry-After" in locked_response.headers

    @pytest.mark.asyncio
    async def test_successful_auth_clears_the_failure_count(self, monkeypatch):
        from starlette.responses import PlainTextResponse

        middleware = self._make_middleware(monkeypatch, auth_succeeds=False)

        async def call_next(request):
            return PlainTextResponse("ok")

        await middleware.dispatch(self._request(), call_next)
        await middleware.dispatch(self._request(), call_next)

        # Switch the same identifier (ip:token prefix) to a successful auth.
        import api.auth.middleware as middleware_module

        def fake_authenticate_token_ok(token, session):
            class _Auth:
                auth_type = "jwt"
                subject = "user1"

            return _Auth()

        monkeypatch.setattr(middleware_module, "authenticate_token", fake_authenticate_token_ok)

        response = await middleware.dispatch(self._request(), call_next)
        assert response.status_code == 200

        # The prior two failures must have been cleared by the success, so a
        # fresh pair of failures should not yet trigger the 3-strike lockout.
        monkeypatch.setattr(middleware_module, "authenticate_token", fake_authenticate_token_ok)
        middleware2 = self._make_middleware(monkeypatch, auth_succeeds=False)

        async def call_next2(request):
            return PlainTextResponse("ok")

        r1 = await middleware2.dispatch(self._request(), call_next2)
        r2 = await middleware2.dispatch(self._request(), call_next2)
        assert r1.status_code == 401
        assert r2.status_code == 401

    @pytest.mark.asyncio
    async def test_locked_out_identifier_is_rejected_before_auth_is_attempted(self, monkeypatch):
        from starlette.responses import PlainTextResponse

        import api.auth.middleware as middleware_module

        middleware = self._make_middleware(monkeypatch, auth_succeeds=False)

        async def call_next(request):
            return PlainTextResponse("ok")

        for _ in range(3):
            await middleware.dispatch(self._request(), call_next)

        calls = {"count": 0}

        def counting_authenticate_token(token, session):
            calls["count"] += 1
            raise ValueError("should not be reached while locked out")

        monkeypatch.setattr(middleware_module, "authenticate_token", counting_authenticate_token)

        response = await middleware.dispatch(self._request(), call_next)

        assert response.status_code == 429
        assert calls["count"] == 0

    @pytest.mark.asyncio
    async def test_different_client_ips_are_locked_out_independently(self, monkeypatch):
        from starlette.responses import PlainTextResponse

        middleware = self._make_middleware(monkeypatch, auth_succeeds=False)

        async def call_next(request):
            return PlainTextResponse("ok")

        for _ in range(3):
            await middleware.dispatch(self._request(client_ip="10.0.0.1"), call_next)

        locked = await middleware.dispatch(self._request(client_ip="10.0.0.1"), call_next)
        still_open = await middleware.dispatch(self._request(client_ip="10.0.0.2"), call_next)

        assert locked.status_code == 429
        assert still_open.status_code == 401


class _NullSession:
    def close(self) -> None:
        pass


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
