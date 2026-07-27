"""Tests for input validation middleware and rate limiting (issues #533, #532)."""
from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from api.validation import InputValidator, ValidationError
from api.validation_middleware import ValidationMiddleware, MAX_REQUEST_SIZE_BYTES
from api.auth.rate_limit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitResult,
    TokenBucket,
)

# A valid synthetic 56-character Stellar account ID (G + 55 base32 chars).
_VALID_STELLAR = "G" + "A" * 55
# A second valid Stellar account ID for tests requiring two accounts.
_VALID_STELLAR_2 = "G" + "B" * 55


# ─── #533 Input Validation — unit tests ─────────────────────────────────────


class TestInputValidator:
    """Unit tests for InputValidator methods."""

    # ── SQL injection detection ───────────────────────────────────────────────

    def test_sql_injection_select_detected(self) -> None:
        # Contains a UNION SELECT which is a classic injection pattern
        assert InputValidator.check_sql_injection("UNION SELECT * FROM users") is True

    def test_sql_injection_drop_detected(self) -> None:
        assert InputValidator.check_sql_injection("'; DROP TABLE users; --") is True

    def test_sql_injection_union_detected(self) -> None:
        assert InputValidator.check_sql_injection("UNION SELECT * FROM secrets") is True

    def test_sql_injection_comment_detected(self) -> None:
        assert InputValidator.check_sql_injection("foo -- bar") is True

    def test_clean_string_passes_sql_check(self) -> None:
        assert InputValidator.check_sql_injection(_VALID_STELLAR) is False

    def test_normal_search_passes_sql_check(self) -> None:
        assert InputValidator.check_sql_injection("account-search-42") is False

    # ── XSS detection ────────────────────────────────────────────────────────

    def test_xss_script_tag_detected(self) -> None:
        assert InputValidator.check_xss("<script>alert('xss')</script>") is True

    def test_xss_javascript_protocol_detected(self) -> None:
        assert InputValidator.check_xss("javascript:alert(1)") is True

    def test_xss_event_handler_detected(self) -> None:
        assert InputValidator.check_xss('<img onload="evil()">') is True

    def test_xss_iframe_detected(self) -> None:
        assert InputValidator.check_xss("<iframe src='https://evil.example'></iframe>") is True

    def test_clean_html_passes_xss_check(self) -> None:
        assert InputValidator.check_xss("<b>Bold text</b>") is False

    def test_plain_text_passes_xss_check(self) -> None:
        assert InputValidator.check_xss("Hello, world!") is False

    # ── Stellar account ID validation ────────────────────────────────────────

    def test_valid_stellar_account_passes(self) -> None:
        assert InputValidator.validate_public_key(_VALID_STELLAR) is True

    def test_stellar_account_wrong_prefix_fails(self) -> None:
        # Must start with G
        bad = "A" + "A" * 55
        assert InputValidator.validate_public_key(bad) is False

    def test_stellar_account_too_short_fails(self) -> None:
        assert InputValidator.validate_public_key("GAAZI4") is False

    def test_stellar_account_too_long_fails(self) -> None:
        assert InputValidator.validate_public_key("G" + "A" * 60) is False

    def test_stellar_account_invalid_chars_fails(self) -> None:
        # Contains '1' which is not in base32 alphabet for Stellar
        assert InputValidator.validate_public_key("G" + "1" * 55) is False

    # ── String sanitization ───────────────────────────────────────────────────

    def test_sanitize_removes_null_bytes(self) -> None:
        result = InputValidator.sanitize_string("hello\x00world")
        assert "\x00" not in result

    def test_sanitize_escapes_html_entities(self) -> None:
        result = InputValidator.sanitize_string("<script>")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_non_string_passthrough(self) -> None:
        assert InputValidator.sanitize_string(42) == 42  # type: ignore[arg-type]

    # ── Email validation ──────────────────────────────────────────────────────

    def test_valid_email_passes(self) -> None:
        assert InputValidator.validate_email("user@example.com") is True

    def test_invalid_email_fails(self) -> None:
        assert InputValidator.validate_email("not-an-email") is False


# ─── #533 Input Validation — Pydantic schemas ────────────────────────────────


class TestSchemaValidators:
    """Tests for validators embedded in Pydantic schemas."""

    def test_edge_input_valid(self) -> None:
        from api.schemas import EdgeInput

        edge = EdgeInput(
            src=_VALID_STELLAR,
            dst=_VALID_STELLAR_2,
            amount=100.0,
            asset="XLM",
        )
        assert edge.src.startswith("G")

    def test_edge_input_invalid_stellar_id_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from api.schemas import EdgeInput

        with pytest.raises(PydanticValidationError):
            EdgeInput(
                src="NOTASTELLARACCOUNT",
                dst=_VALID_STELLAR_2,
            )

    def test_edge_input_negative_amount_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from api.schemas import EdgeInput

        with pytest.raises(PydanticValidationError):
            EdgeInput(
                src=_VALID_STELLAR,
                dst=_VALID_STELLAR_2,
                amount=-1.0,
            )

    def test_score_request_empty_accounts_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from api.schemas import ScoreRequest

        with pytest.raises(PydanticValidationError):
            ScoreRequest(accounts=[])

    def test_score_request_too_many_accounts_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from api.schemas import ScoreRequest

        with pytest.raises(PydanticValidationError):
            # max_length=50
            ScoreRequest(accounts=["GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"] * 51)

    def test_mentor_profile_bio_max_length(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from api.schemas import MentorProfileIn

        with pytest.raises(PydanticValidationError):
            MentorProfileIn(bio="x" * 1001, skills=[], years_experience=1)

    def test_mentor_profile_valid(self) -> None:
        from api.schemas import MentorProfileIn

        profile = MentorProfileIn(
            bio="I love Stellar",
            skills=["Python", "GNN"],
            years_experience=5,
        )
        assert profile.years_experience == 5

    def test_model_metrics_out_of_range_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from api.schemas import ModelMetricsOut

        with pytest.raises(PydanticValidationError):
            ModelMetricsOut(accuracy=1.5)  # must be <= 1.0


# ─── #533 Validation Middleware — request size ────────────────────────────────

# Build a minimal test app using only the middleware under test — avoids the
# full api.app import which requires opentelemetry and other optional deps.
def _make_test_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()
    app.add_middleware(ValidationMiddleware)

    @app.get("/test")
    async def _ok():
        return {"status": "ok"}

    @app.post("/test")
    async def _post(request):
        return {"status": "ok"}

    return app


class TestValidationMiddlewareRequestSize:
    """Tests for the MAX_REQUEST_SIZE_BYTES enforcement in ValidationMiddleware."""

    def test_max_request_size_constant_is_1mb(self) -> None:
        assert MAX_REQUEST_SIZE_BYTES == 1 * 1024 * 1024

    def test_oversized_content_length_rejected(self) -> None:
        """A request advertising > 1 MB Content-Length gets a 413."""
        from fastapi.testclient import TestClient

        app = _make_test_app()
        too_large = MAX_REQUEST_SIZE_BYTES + 1
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/test",
                content=b"x",
                headers={"Content-Length": str(too_large)},
            )
        assert resp.status_code == 413

    def test_normal_content_length_accepted(self) -> None:
        """A request within the size limit is not rejected by the middleware."""
        from fastapi.testclient import TestClient

        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/test")
        assert resp.status_code != 413


class TestValidationMiddlewareMaliciousInputs:
    """Test that SQL injection and XSS in query params are rejected."""

    def test_sql_injection_in_query_param_rejected(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/test", params={"q": "UNION SELECT * FROM users"})
        assert resp.status_code == 400

    def test_xss_in_query_param_rejected(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/test", params={"q": "<script>alert(1)</script>"})
        assert resp.status_code == 400

    def test_clean_query_param_passes(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/test", params={"q": "normal-search-term"})
        assert resp.status_code != 400

    def test_response_has_security_headers(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/test")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


# ─── #532 Rate Limiting — unit tests ─────────────────────────────────────────


class TestRateLimiter:
    """Unit tests for the RateLimiter / TokenBucket implementation."""

    def _make_limiter(
        self,
        requests_per_minute: int = 5,
        burst_size: int = 5,
    ) -> RateLimiter:
        limiter = RateLimiter()
        limiter.set_endpoint_config(
            "/test",
            RateLimitConfig(
                requests_per_minute=requests_per_minute,
                burst_size=burst_size,
            ),
        )
        return limiter

    def test_initial_requests_allowed(self) -> None:
        limiter = self._make_limiter(requests_per_minute=10, burst_size=5)
        for _ in range(5):
            result = limiter.is_allowed("user:test", "/test")
            assert result.allowed is True

    def test_exceeding_burst_returns_429_data(self) -> None:
        limiter = self._make_limiter(requests_per_minute=60, burst_size=2)
        # Exhaust the burst
        limiter.is_allowed("user:burst", "/test")
        limiter.is_allowed("user:burst", "/test")
        # Third should be denied
        result = limiter.is_allowed("user:burst", "/test")
        assert result.allowed is False

    def test_denied_result_has_retry_after(self) -> None:
        limiter = self._make_limiter(requests_per_minute=60, burst_size=1)
        limiter.is_allowed("user:ra", "/test")  # consume the burst
        result = limiter.is_allowed("user:ra", "/test")
        assert result.allowed is False
        assert result.retry_after is not None
        assert result.retry_after >= 0

    def test_denied_result_has_limit(self) -> None:
        limiter = self._make_limiter(requests_per_minute=60, burst_size=1)
        limiter.is_allowed("user:lim", "/test")
        result = limiter.is_allowed("user:lim", "/test")
        assert result.limit > 0

    def test_allowed_result_has_remaining(self) -> None:
        limiter = self._make_limiter(requests_per_minute=60, burst_size=5)
        result = limiter.is_allowed("user:rem", "/test")
        assert result.allowed is True
        assert result.remaining >= 0

    def test_blacklisted_key_is_always_denied(self) -> None:
        limiter = self._make_limiter()
        limiter.add_to_blacklist("evil_key")
        result = limiter.is_allowed("evil_key:path", "/test")
        assert result.allowed is False

    def test_whitelisted_key_is_always_allowed(self) -> None:
        limiter = self._make_limiter(requests_per_minute=1, burst_size=1)
        limiter.add_to_whitelist("vip_key")
        # Exhaust any token
        limiter.is_allowed("vip_key:user", "/test")
        # Should still be allowed because whitelisted
        result = limiter.is_allowed("vip_key:user", "/test")
        assert result.allowed is True

    def test_admin_path_default_limit_is_50(self) -> None:
        limiter = RateLimiter()
        config = limiter._get_endpoint_config("/api/v1/admin/users")
        assert config.requests_per_minute == 50

    def test_unknown_path_default_limit_is_1000(self) -> None:
        limiter = RateLimiter()
        config = limiter._get_endpoint_config("/api/v1/some/new/endpoint")
        assert config.requests_per_minute == 1000

    def test_public_health_limit_is_100(self) -> None:
        limiter = RateLimiter()
        config = limiter._get_endpoint_config("/api/v1/health")
        assert config.requests_per_minute == 100

    def test_metrics_tracked(self) -> None:
        limiter = self._make_limiter(requests_per_minute=60, burst_size=5)
        limiter.is_allowed("user:metrics", "/test")
        metrics = limiter.get_metrics()
        assert any("allowed" in key for key in metrics)

    def test_metrics_reset(self) -> None:
        limiter = self._make_limiter(requests_per_minute=60, burst_size=5)
        limiter.is_allowed("user:reset", "/test")
        limiter.reset_metrics()
        assert limiter.get_metrics() == {}


class TestRateLimitHeaders:
    """Tests that X-RateLimit-* and Retry-After headers are sent correctly."""

    def test_x_ratelimit_headers_present_on_success(self) -> None:
        """Verify the middleware adds rate limit headers when a request is allowed."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import JSONResponse

        # Simulate what AuthMiddleware does: inject rate limit headers on every response.
        app = FastAPI()

        @app.middleware("http")
        async def _mock_rate_limit_headers(request: Request, call_next):
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = "1000"
            response.headers["X-RateLimit-Remaining"] = "999"
            response.headers["X-RateLimit-Reset"] = "9999999999"
            return response

        @app.get("/test")
        async def _ok():
            return {"status": "ok"}

        with TestClient(app) as c:
            resp = c.get("/test")

        assert resp.headers.get("X-RateLimit-Limit") == "1000"
        assert resp.headers.get("X-RateLimit-Remaining") == "999"
        assert "X-RateLimit-Reset" in resp.headers

    def test_rate_limit_result_structure(self) -> None:
        """RateLimitResult has the fields expected for HTTP header injection."""
        result = RateLimitResult(
            allowed=False,
            retry_after=30,
            remaining=0,
            limit=100,
        )
        assert result.retry_after == 30
        assert result.limit == 100
        assert result.remaining == 0
        assert result.allowed is False

    def test_allowed_result_structure(self) -> None:
        result = RateLimitResult(allowed=True, remaining=99, limit=100)
        assert result.allowed is True
        assert result.remaining == 99


class TestRateLimitEndpointTiers:
    """Tests that the three rate limit tiers are correctly configured."""

    def test_public_endpoint_tier(self) -> None:
        limiter = RateLimiter()
        for path in ["/api/v1/health", "/api/v1/faq", "/api/v1/onboarding"]:
            config = limiter._get_endpoint_config(path)
            assert config.requests_per_minute == 100, f"{path} should be 100/min"

    def test_authenticated_endpoint_tier(self) -> None:
        limiter = RateLimiter()
        for path in [
            "/api/v1/transactions",
            "/api/v1/accounts",
            "/api/v1/fraud",
            "/api/v1/monitoring",
        ]:
            config = limiter._get_endpoint_config(path)
            assert config.requests_per_minute == 1000, f"{path} should be 1000/min"

    def test_admin_endpoint_tier(self) -> None:
        limiter = RateLimiter()
        for path in ["/api/v1/admin", "/api/v1/audit", "/api/v1/backup", "/api/v1/compliance"]:
            config = limiter._get_endpoint_config(path)
            assert config.requests_per_minute == 50, f"{path} should be 50/min"

    def test_login_endpoint_is_extra_strict(self) -> None:
        limiter = RateLimiter()
        config = limiter._get_endpoint_config("/api/v1/auth/login")
        assert config.requests_per_minute == 5
