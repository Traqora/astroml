"""Authentication configuration (issue #240)."""
from __future__ import annotations

import os

SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or os.environ.get(
    "SECRET_KEY", "change-me-in-production"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.environ.get("ACCESS_TOKEN_EXPIRE_HOURS", "24"))
API_KEY_EXPIRE_DAYS = int(os.environ.get("API_KEY_EXPIRE_DAYS", "365"))

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() in ("1", "true", "yes")


def is_auth_enabled() -> bool:
    """Read AUTH_ENABLED at call time (supports test monkeypatching)."""
    return os.environ.get("AUTH_ENABLED", "true").lower() in ("1", "true", "yes")

DEFAULT_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# Rate limiting configuration
JWT_RATE_LIMIT_PER_MINUTE = int(os.environ.get("JWT_RATE_LIMIT_PER_MINUTE", "100"))
API_KEY_RATE_LIMIT_PER_MINUTE = int(os.environ.get("API_KEY_RATE_LIMIT_PER_MINUTE", "1000"))

# Redis configuration for distributed rate limiting
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_RATE_LIMIT_ENABLED = os.environ.get("REDIS_RATE_LIMIT_ENABLED", "false").lower() in ("1", "true", "yes")

# Admin override configuration
ADMIN_OVERRIDE_ENABLED = os.environ.get("ADMIN_OVERRIDE_ENABLED", "true").lower() in ("1", "true", "yes")
ADMIN_WHITELIST = os.environ.get("ADMIN_WHITELIST", "").split(",") if os.environ.get("ADMIN_WHITELIST") else []
ADMIN_BLACKLIST = os.environ.get("ADMIN_BLACKLIST", "").split(",") if os.environ.get("ADMIN_BLACKLIST") else []

# Rate limit algorithm: 'token_bucket' or 'sliding_window'
RATE_LIMIT_ALGORITHM = os.environ.get("RATE_LIMIT_ALGORITHM", "token_bucket")

# Sliding window configuration
SLIDING_WINDOW_SIZE_SECONDS = int(os.environ.get("SLIDING_WINDOW_SIZE_SECONDS", "60"))

PUBLIC_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/v1/auth/login",
})