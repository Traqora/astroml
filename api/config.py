"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "sqlite+aiosqlite:///./astroml.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_version: str = "1.0.0"

    # CORS — deny all by default; set CORS_ORIGINS in production/dev .env
    # e.g. CORS_ORIGINS=https://app.example.com,https://admin.example.com
    cors_origins: list[str] = []
    cors_credentials: bool = False

    @model_validator(mode="after")
    def _cors_wildcard_credentials_guard(self) -> "Settings":
        if "*" in self.cors_origins and self.cors_credentials:
            raise ValueError(
                "cors_origins=['*'] combined with cors_credentials=True is not allowed. "
                "Browsers reject credentialed requests to wildcard origins; "
                "set explicit origins or disable cors_credentials."
            )
        return self

    # Content Security Policy
    csp_report_only: bool = True  # Use report-only mode in development
    csp_report_uri: str | None = None  # URI to send CSP violation reports
    csp_enable_nonce: bool = True  # Generate nonce for script-src

    # HTTPS Enforcement
    https_enabled: bool = False  # Enable HTTPS redirects (production only)
    https_allowed_hosts: list[str] = []  # Allowed hostnames for HTTPS
    hsts_enabled: bool = False  # Enable HSTS headers (production only)
    hsts_max_age: int = 31536000  # HSTS max-age in seconds (1 year)
    hsts_include_subdomains: bool = True  # Apply HSTS to subdomains
    hsts_preload: bool = False  # Include in browser preload list
    secure_proxy_ssl_header: tuple[str, str] | None = (
        None  # For load balancers: ("X-Forwarded-Proto", "https")
    )

    # Auth
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60

    # ML model paths
    model_path: str = "outputs/model.pkl"
    benchmark_results_dir: str = "benchmark_results"

    # Logging
    log_level: str = "INFO"

    # Distributed Tracing (issue #336)
    tracing_enabled: bool = False
    tracing_exporter: str = "jaeger"  # jaeger|zipkin|console
    tracing_sample_rate: float = 0.1  # 10% sampling by default
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"
    service_name: str = "astroml-api"

    # Contact form / support tickets (issue #305)
    # Empty reCAPTCHA secret disables spam verification (dev/test default).
    contact_recaptcha_secret: str = ""
    contact_recaptcha_min_score: float = 0.5  # reCAPTCHA v3 score threshold
    # Empty SendGrid key → emails are logged instead of sent (dev/test default).
    sendgrid_api_key: str = ""
    contact_email_from: str = "no-reply@astroml.dev"
    contact_support_email: str = "support@astroml.dev"

    # Feedback collection / GitHub integration (issue #308)
    # When both are set, new feedback opens a GitHub issue in this repo.
    github_token: str = ""
    github_repo: str = ""  # "owner/repo"

    # LLM Settings (issue #440)
    llm_provider: str = "openai"
    llm_encryption_key: str = "change-me-in-production-llm-key-32b"
    db_query_timeout_seconds: int = Field(5, ge=1, le=300)
    external_http_timeout_seconds: int = Field(10, ge=1, le=300)  # TODO(#1231): Wire to actual outbound HTTP calls when added to V1 endpoints. Any V1 endpoint that makes outbound HTTP calls MUST use settings.external_http_timeout_seconds at that call site and add a corresponding test.
    llm_timeout_seconds: int = Field(30, ge=1, le=300)

    # Request / pagination limits (issue #1247)
    max_request_body_size: int = 10 * 1024 * 1024  # 10 MB; env var MAX_REQUEST_BODY_SIZE
    max_page_size: int = 100  # env var MAX_PAGE_SIZE
    default_page_size: int = 20  # env var DEFAULT_PAGE_SIZE


settings = Settings()
