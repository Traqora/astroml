"""Configuration for E2E tests."""
import os
from pathlib import Path

# Test environment
TEST_ENV = os.getenv("TEST_ENV", "local")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/astroml_test")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Test configuration
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "30"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
SKIP_SLOW_TESTS = os.getenv("E2E_SKIP_SLOW_TESTS", "0") == "1"

# Report paths
REPORT_DIR = Path(os.getenv("REPORT_DIR", "./test-results"))
REPORT_DIR.mkdir(exist_ok=True)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# GitHub settings
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "Traqora")
GITHUB_REPO = os.getenv("GITHUB_REPO", "astroml")
