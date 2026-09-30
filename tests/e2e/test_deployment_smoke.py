"""Post-deployment smoke tests for issue #560.

Verifies that the deployment succeeded by checking:
- Health check endpoint
- API root endpoint
- Database connectivity
- Cache connectivity

These tests run after deployment with a 5-minute timeout.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest
import requests


class TestDeploymentSmoke:
    """Smoke tests to verify deployment success."""

    BASE_URL = "http://localhost:8000"
    TIMEOUT = 300  # 5 minutes total timeout

    def test_health_check(self):
        """Test that health check endpoint returns 200 (issue #560)."""
        response = requests.get(
            f"{self.BASE_URL}/healthz",
            timeout=30,
        )
        assert response.status_code == 200, f"Health check failed: {response.text}"
        data = response.json()
        assert "status" in data, "Health check response missing 'status' field"
        assert data["status"] == "healthy", f"Health check status not healthy: {data}"

    def test_api_root(self):
        """Test that API root returns app info (issue #560)."""
        response = requests.get(
            f"{self.BASE_URL}/",
            timeout=30,
        )
        assert response.status_code == 200, f"API root failed: {response.text}"
        data = response.json()
        assert "app_name" in data or "name" in data, "API root response missing app name"
        assert "version" in data, "API root response missing version"

    def test_database_connectivity(self):
        """Test database connectivity via health check (issue #560)."""
        response = requests.get(
            f"{self.BASE_URL}/healthz/db",
            timeout=30,
        )
        assert response.status_code == 200, f"DB health check failed: {response.text}"
        data = response.json()
        assert "status" in data, "DB health check response missing 'status' field"
        assert data["status"] == "healthy", f"DB health check status not healthy: {data}"

    def test_cache_connectivity(self):
        """Test cache connectivity via health check (issue #560)."""
        response = requests.get(
            f"{self.BASE_URL}/healthz/cache",
            timeout=30,
        )
        assert response.status_code == 200, f"Cache health check failed: {response.text}"
        data = response.json()
        assert "status" in data, "Cache health check response missing 'status' field"
        assert data["status"] == "healthy", f"Cache health check status not healthy: {data}"

    def test_api_endpoints_responsive(self):
        """Test that key API endpoints are responsive (issue #560)."""
        endpoints = [
            "/api/v1/healthz",
            "/api/v1/models",
        ]

        for endpoint in endpoints:
            response = requests.get(
                f"{self.BASE_URL}{endpoint}",
                timeout=30,
            )
            # Accept 200 or 401 (auth required) as success
            assert response.status_code in [
                200,
                401,
            ], f"Endpoint {endpoint} failed with status {response.status_code}: {response.text}"

    def test_deployment_timestamp(self):
        """Verify deployment is recent (within last hour)."""
        response = requests.get(
            f"{self.BASE_URL}/healthz",
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()

        # Check if deployment timestamp is present and recent
        if "deployment_time" in data:
            deployment_time = data["deployment_time"]
            current_time = time.time()
            time_diff = current_time - deployment_time
            assert time_diff < 3600, f"Deployment timestamp is too old: {time_diff}s ago"


class TestDeploymentSmokeConfigurable:
    """Configurable smoke tests with custom base URL."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize with custom base URL.

        Args:
            base_url: Base URL for the deployed application
        """
        self.base_url = base_url

    def run_all_smoke_tests(self) -> Dict[str, bool]:
        """Run all smoke tests and return results.

        Returns:
            Dictionary mapping test names to success status
        """
        results = {}

        tests = [
            ("health_check", self._test_health_check),
            ("api_root", self._test_api_root),
            ("database", self._test_database),
            ("cache", self._test_cache),
        ]

        for test_name, test_func in tests:
            try:
                test_func()
                results[test_name] = True
            except Exception as e:
                results[test_name] = False
                print(f"Test {test_name} failed: {e}")

        return results

    def _test_health_check(self):
        """Test health check."""
        response = requests.get(f"{self.base_url}/healthz", timeout=30)
        assert response.status_code == 200

    def _test_api_root(self):
        """Test API root."""
        response = requests.get(f"{self.base_url}/", timeout=30)
        assert response.status_code == 200

    def _test_database(self):
        """Test database connectivity."""
        response = requests.get(f"{self.base_url}/healthz/db", timeout=30)
        assert response.status_code == 200

    def _test_cache(self):
        """Test cache connectivity."""
        response = requests.get(f"{self.base_url}/healthz/cache", timeout=30)
        assert response.status_code == 200


def run_smoke_tests(base_url: str = "http://localhost:8000") -> bool:
    """Run smoke tests programmatically.

    Args:
        base_url: Base URL for the deployed application

    Returns:
        True if all tests pass, False otherwise
    """
    tester = TestDeploymentSmokeConfigurable(base_url)
    results = tester.run_all_smoke_tests()

    all_passed = all(results.values())

    if not all_passed:
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"Smoke tests failed: {failed_tests}")

    return all_passed


if __name__ == "__main__":
    """Run smoke tests when executed directly."""
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Running smoke tests against {base_url}...")
    success = run_smoke_tests(base_url)

    if success:
        print("All smoke tests passed!")
        sys.exit(0)
    else:
        print("Smoke tests failed!")
        sys.exit(1)
