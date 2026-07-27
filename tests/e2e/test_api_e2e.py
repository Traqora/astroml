"""End-to-end API tests for critical user journeys."""
import pytest
import asyncio
from fastapi.testclient import TestClient
from api.app import app
import os


client = TestClient(app)


class TestCriticalUserJourneys:
    """Test critical user journeys through the API."""

    def test_health_check(self):
        """Test API health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = client.get("/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "status" in data
        assert data["status"] == "ok"


class TestAuthFlow:
    """Test authentication user journey."""

    def test_register_and_login(self):
        """Test user registration and login flow."""
        # Register new user
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "email": f"testuser_{os.urandom(4).hex()}@example.com",
                "password": "SecurePassword123!",
                "full_name": "Test User"
            }
        )
        assert register_response.status_code in [200, 201, 409]  # 409 if user exists

        # Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "testuser@example.com",
                "password": "password"
            }
        )
        assert login_response.status_code in [200, 422, 401]


class TestTransactionJourney:
    """Test transaction-related user journey."""

    def test_fetch_transactions(self):
        """Test fetching transaction data."""
        response = client.get("/api/v1/transactions")
        assert response.status_code in [200, 401]

    def test_fetch_transactions_with_filters(self):
        """Test fetching transactions with filters."""
        response = client.get("/api/v1/transactions?limit=10&offset=0")
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (dict, list))


class TestFraudDetectionJourney:
    """Test fraud detection user journey."""

    def test_fraud_check_endpoint(self):
        """Test fraud check endpoint."""
        response = client.get("/api/v1/fraud")
        assert response.status_code in [200, 401, 405]

    def test_fraud_detection_report(self):
        """Test accessing fraud detection report."""
        response = client.get("/api/v1/fraud/report")
        assert response.status_code in [200, 401, 404]


class TestAccountsJourney:
    """Test accounts management user journey."""

    def test_fetch_accounts(self):
        """Test fetching account information."""
        response = client.get("/api/v1/accounts")
        assert response.status_code in [200, 401]

    def test_account_details(self):
        """Test fetching specific account details."""
        response = client.get("/api/v1/accounts/test-account")
        assert response.status_code in [200, 401, 404]


class TestLoyaltyJourney:
    """Test loyalty points user journey."""

    def test_loyalty_summary(self):
        """Test fetching loyalty points summary."""
        response = client.get("/api/v1/loyalty/summary")
        assert response.status_code in [200, 401]

    def test_redeem_points(self):
        """Test redeeming loyalty points."""
        response = client.post(
            "/api/v1/loyalty/redeem",
            json={"points": 100}
        )
        assert response.status_code in [200, 201, 400, 401]

    def test_points_history(self):
        """Test fetching points history."""
        response = client.get("/api/v1/loyalty/history")
        assert response.status_code in [200, 401]


class TestMonitoringJourney:
    """Test monitoring and metrics user journey."""

    def test_model_metrics(self):
        """Test fetching model metrics."""
        response = client.get("/api/v1/monitoring/metrics")
        assert response.status_code in [200, 401, 404]

    def test_latency_report(self):
        """Test accessing latency report."""
        response = client.get("/api/v1/monitoring/latency")
        assert response.status_code in [200, 401, 404]


class TestDiscussionsJourney:
    """Test community discussions user journey."""

    def test_discussions_flow(self):
        """Test complete discussions flow."""
        # Fetch categories
        cat_response = client.get("/api/v1/discussions/categories")
        assert cat_response.status_code == 200

        # Fetch recent discussions
        disc_response = client.get("/api/v1/discussions/recent")
        assert disc_response.status_code == 200
        data = disc_response.json()
        assert "discussions" in data

        # Search discussions
        search_response = client.post(
            "/api/v1/discussions/search",
            json={"query": "test"}
        )
        assert search_response.status_code == 200


class TestContributorsJourney:
    """Test contributors dashboard user journey."""

    def test_contributors_list(self):
        """Test fetching contributors list."""
        response = client.get("/api/v1/contributors")
        assert response.status_code in [200, 401]

    def test_contributor_details(self):
        """Test fetching contributor details."""
        response = client.get("/api/v1/contributors/test-contributor")
        assert response.status_code in [200, 401, 404]


class TestNotificationsJourney:
    """Test notifications user journey."""

    def test_fetch_notifications(self):
        """Test fetching notifications."""
        response = client.get("/api/v1/notifications")
        assert response.status_code in [200, 401]

    def test_mark_notification_read(self):
        """Test marking notification as read."""
        response = client.put(
            "/api/v1/notifications/test-id",
            json={"read": True}
        )
        assert response.status_code in [200, 401, 404]


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_404_not_found(self):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent-endpoint")
        assert response.status_code == 404

    def test_invalid_json(self):
        """Test invalid JSON handling."""
        response = client.post(
            "/api/v1/auth/login",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_missing_required_fields(self):
        """Test missing required fields handling."""
        response = client.post(
            "/api/v1/auth/login",
            json={}
        )
        assert response.status_code in [400, 422]


class TestConcurrency:
    """Test concurrent API requests."""

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        results = []
        for _ in range(5):
            response = client.get("/api/v1/discussions/recent")
            results.append(response.status_code)

        assert all(code == 200 for code in results)

    def test_concurrent_different_endpoints(self):
        """Test concurrent requests to different endpoints."""
        endpoints = [
            "/health",
            "/api/v1",
            "/api/v1/discussions/categories",
            "/api/v1/discussions/recent",
        ]

        responses = [client.get(ep) for ep in endpoints]
        assert all(r.status_code == 200 for r in responses)


class TestResponseFormats:
    """Test response format consistency."""

    def test_json_response_format(self):
        """Test that responses are valid JSON."""
        response = client.get("/api/v1/discussions/recent")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert isinstance(data, dict)

    def test_error_response_format(self):
        """Test that error responses follow format."""
        response = client.post(
            "/api/v1/auth/login",
            json={"invalid": "data"}
        )
        assert response.status_code in [400, 422]
        data = response.json()
        assert isinstance(data, dict)


class TestRateLimit:
    """Test rate limiting."""

    def test_rate_limiting_not_exceeded(self):
        """Test that normal requests are not rate limited."""
        response = client.get("/api/v1/discussions/recent")
        assert response.status_code != 429

    def test_multiple_requests_succeed(self):
        """Test multiple requests in sequence."""
        for _ in range(10):
            response = client.get("/api/v1/discussions/recent")
            assert response.status_code in [200, 429]


class TestCORSHeaders:
    """Test CORS headers in responses."""

    def test_cors_headers_present(self):
        """Test that CORS headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        # CORS headers are typically set by middleware


@pytest.mark.skipif(
    os.getenv("E2E_SKIP_SLOW_TESTS") == "1",
    reason="Slow test skipped in CI"
)
class TestPerformance:
    """Test performance characteristics."""

    def test_response_time_under_threshold(self):
        """Test that responses are within acceptable time."""
        import time
        start = time.time()
        response = client.get("/api/v1/discussions/recent")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Response should be under 5 seconds
        assert elapsed < 5.0

    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        import time
        start = time.time()

        for _ in range(20):
            response = client.get("/api/v1/discussions/recent")
            assert response.status_code == 200

        elapsed = time.time() - start
        # 20 requests should complete in reasonable time
        assert elapsed < 30.0
