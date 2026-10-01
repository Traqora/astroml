"""Database session factory for AstroML.

This module provides database connection management and session creation with:
- Configuration loading from YAML files
- Environment variable override support
- Connection pooling with health monitoring
- Query profiling in debug mode
- Pool statistics and health checks
- A standard pagination envelope for offset-paginated query results (#948)

Key components:
- DatabaseConfig: Validated database configuration
- get_engine: Cached SQLAlchemy engine
- get_session: Session factory
- load_database_config: YAML configuration loader
- resolve_database_url: URL resolution with fallbacks
- PageParams / Page: Standard pagination request/response envelope
- paginate_offset: Compute an envelope from a 1-based page + total row count

Dependencies:
- sqlalchemy: ORM and database toolkit
- pydantic: Configuration validation
- pyyaml: YAML parsing
"""

from __future__ import annotations

import logging
import os
import pathlib
from functools import lru_cache
from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from astroml.db.pool_health import PoolStats
    from astroml.observability.health import CheckResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseConfig(BaseModel):
    """Database configuration with validation."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="astroml", min_length=1, description="Database name")
    user: str = Field(default="astroml", min_length=1, description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout seconds")
    pool_recycle: int = Field(
        default=1800, ge=-1, description="Pool connection recycle seconds (-1 disables)"
    )

    @property
    def max_connections(self) -> int:
        """Upper bound on concurrent connections this engine may open."""
        return self.pool_size + self.max_overflow

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host is not empty."""
        if not v or not v.strip():
            raise ValueError("Database host cannot be empty")
        return v.strip()

    @field_validator("name", "user")
    @classmethod
    def validate_not_blank(cls, v: str, info: ValidationInfo) -> str:
        """Reject whitespace-only values and strip surrounding whitespace.

        ``min_length=1`` alone lets a value like ``"   "`` through, which would
        silently produce a malformed connection URL (issue #977) instead of
        failing validation the way an empty ``host`` already does.
        """
        if not v.strip():
            raise ValueError(f"Database {info.field_name} cannot be blank")
        return v.strip()

    def to_url(self) -> str:
        """Convert configuration to PostgreSQL URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def masked_url(self) -> str:
        """Connection URL with the password masked, safe for logs/diagnostics.

        Unlike :meth:`to_url`, never emits credentials in clear text
        (see CodeQL rule py/clear-text-logging-sensitive-data).
        """
        masked = "***" if self.password else ""
        return f"postgresql://{self.user}:{masked}@{self.host}:{self.port}/{self.name}"

    @classmethod
    def from_dict(cls, data: dict) -> DatabaseConfig:
        """Create configuration from dictionary with validation."""
        return cls(**data)


def load_database_config(config_path: pathlib.Path | None = None) -> DatabaseConfig:
    """Load and validate database configuration from YAML file.

    Args:
        config_path: Path to database.yaml. Defaults to config/database.yaml.

    Returns:
        Validated DatabaseConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config is invalid.
    """
    if config_path is None:
        config_path = pathlib.Path("config/database.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Database config file not found at {config_path}. "
            f"Please create it or set ASTROML_DATABASE_URL environment variable."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # #151 — Surface clear, schema-pointing errors instead of silently
    # falling back to defaults when the YAML is malformed or missing the
    # `database:` root.
    if cfg is None:
        raise ValueError(f"{config_path} is empty. Expected:\n{_database_yaml_template()}")
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{config_path} must be a YAML mapping at the top level "
            f"(got {type(cfg).__name__}). Expected:\n{_database_yaml_template()}"
        )
    if "database" not in cfg:
        raise ValueError(
            f"{config_path} is missing the `database:` key. Expected:\n"
            f"{_database_yaml_template()}"
        )
    if not isinstance(cfg["database"], dict):
        raise ValueError(
            f"`database:` in {config_path} must be a mapping "
            f"(got {type(cfg['database']).__name__}). Expected:\n"
            f"{_database_yaml_template()}"
        )

    try:
        return DatabaseConfig.from_dict(cfg["database"])
    except ValidationError as e:
        raise ValueError(
            f"Invalid database configuration in {config_path}:\n"
            f"{e}\n\nExpected schema:\n{_database_yaml_template()}"
        ) from e


def _database_yaml_template() -> str:
    """Schema-by-example printed in error messages.

    Mirrors config/database.yaml so operators can copy-paste a known-good block.

    Returns:
        YAML template string for database configuration.
    """
    return (
        "database:\n"
        "  host: localhost            # non-empty string\n"
        "  port: 5432                 # 1..65535\n"
        "  name: astroml              # non-empty string\n"
        "  user: astroml              # non-empty string\n"
        '  password: ""               # string, may be empty\n'
        "  pool_size: 10              # int\n"
        "  max_overflow: 20           # int\n"
        "  pool_timeout: 30           # int\n"
        "  pool_recycle: 1800         # int\n"
    )


def resolve_database_url() -> str:
    """Return the database URL, preferring env var over config file."""
    env_url = os.environ.get("ASTROML_DATABASE_URL")
    if env_url:
        return env_url

    try:
        config = load_database_config()
        return config.to_url()
    except FileNotFoundError:
        # Fall back to default if config doesn't exist
        return "postgresql://astroml:@localhost:5432/astroml"
    except (ValidationError, ValueError) as e:
        # Re-raise validation errors with clear message. `load_database_config`
        # now raises ValueError (with schema-by-example), but legacy callers
        # may still see pydantic ValidationError if a future schema check
        # bypasses the wrapper — catch both.
        raise ValueError(
            f"Database configuration error: {e}\n"
            f"Please fix config/database.yaml or set ASTROML_DATABASE_URL environment variable."
        ) from e


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine."""
    try:
        config = load_database_config()
        engine = create_engine(
            resolve_database_url(),
            pool_pre_ping=True,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
        )
    except Exception as e:
        # Issue #970 — this previously swallowed the error silently, so a
        # malformed config/database.yaml (as opposed to the expected "no
        # config file in this environment" case) would fail over to default
        # pool settings with no trace of why. Log it with the original
        # error before falling back so misconfigurations are still visible.
        logger.warning(
            "Failed to load database config (%s). Falling back to default pool settings",
            e,
        )
        engine = create_engine(
            resolve_database_url(),
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800,
        )

    # Enable query profiling in debug mode
    _enable_query_profiling_if_debug(engine)

    return engine


def _enable_query_profiling_if_debug(engine: Engine) -> None:
    """Enable query profiling if ASTROML_DEBUG is set.

    Args:
        engine: SQLAlchemy engine to profile
    """
    if os.environ.get("ASTROML_DEBUG", "").lower() in ("true", "1", "yes"):
        try:
            from astroml.db.query_profiler import configure_query_logging

            configure_query_logging(
                log_level="DEBUG",
                enable_profiling=True,
                slow_query_threshold_ms=int(
                    os.environ.get("ASTROML_SLOW_QUERY_THRESHOLD_MS", "100")
                ),
            )
            logger.info("Query profiling enabled in debug mode")
        except ImportError:
            logger.warning("Query profiler module not available")


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return a cached session factory bound to the shared engine.

    Issue #982 — building a ``sessionmaker`` per call is wasted work on hot
    paths; the factory is created once and reused. Call
    ``get_session_factory.cache_clear()`` alongside ``get_engine.cache_clear()``
    when the engine is rebuilt.
    """
    return sessionmaker(bind=get_engine())


def get_session() -> Session:
    """Return a new SQLAlchemy session from the cached factory."""
    return get_session_factory()()


def get_pool_stats() -> PoolStats:
    """Return a snapshot of the shared engine's connection pool (#550)."""
    from astroml.db.pool_health import collect_pool_stats

    return collect_pool_stats(get_engine())


def check_connection_pool() -> CheckResult:
    """Evaluate connection pool health for the shared engine (#550).

    Returns:
        A ``CheckResult`` named ``"db_pool"`` — ``DEGRADED`` once utilization
        reaches 80% of capacity, ``FAIL`` when the pool is exhausted.
    """
    from astroml.db.pool_health import check_pool

    return check_pool(get_engine())


# ---------------------------------------------------------------------------
# Pagination envelope standard (#948)
# ---------------------------------------------------------------------------
#
# ``astroml/api/routers/accounts.py`` and ``astroml/api/routers/fraud.py``
# each hand-roll the same ``(items, total, page, page_size)`` response shape
# and the same ``offset = (page - 1) * page_size`` math for every paginated
# endpoint. ``PageParams``/``Page``/``paginate_offset`` below give routers
# (and any other offset-paginated query site, e.g. future CLI/report
# tooling) one canonical envelope and one canonical offset computation
# instead of each call site reimplementing both.


class PageParams(BaseModel):
    """Validated 1-based page request parameters.

    Mirrors the ``page``/``page_size`` query parameters already used by the
    account/fraud routers (1-based page numbers, page size capped at 100).
    """

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        """Zero-based row offset for this page, e.g. for ``.offset(...)``."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Row limit for this page, e.g. for ``.limit(...)``."""
        return self.page_size


class Page(BaseModel, Generic[T]):
    """Standard paginated response envelope.

    ``items`` holds the current page's rows; ``total`` is the full matching
    row count (independent of the page window), used to compute
    ``total_pages`` and ``has_next``/``has_previous`` for the client.
    """

    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)

    @property
    def total_pages(self) -> int:
        """Total number of pages for ``total`` rows at this ``page_size``.

        ``0`` when there are no matching rows at all, matching the
        convention that an empty result set has no pages rather than one
        empty page.
        """
        if self.total == 0:
            return 0
        return -(-self.total // self.page_size)  # ceil division without floats

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        return self.page > 1


def paginate_offset(items: Sequence[T], total: int, params: PageParams) -> Page[T]:
    """Build a :class:`Page` envelope from an already-fetched page of rows.

    ``items`` should already be the page-window slice returned by a query
    using ``params.offset``/``params.limit`` (or the equivalent
    ``.offset()``/``.limit()`` calls) — this function only assembles the
    envelope and derived page metadata, it does not query the database
    itself, so it works the same for sync and async sessions.

    Args:
        items: The rows for the requested page (already offset/limited).
        total: Total matching row count across all pages.
        params: The validated page request used to fetch ``items``.

    Returns:
        A :class:`Page` envelope combining ``items`` with pagination
        metadata derived from ``total`` and ``params``.
    """
    return Page[T](items=list(items), total=total, page=params.page, page_size=params.page_size)
