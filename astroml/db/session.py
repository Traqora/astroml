"""Database session factory for AstroML.

Resolves the database URL from (in priority order):
1. ``ASTROML_DATABASE_URL`` environment variable
2. ``config/database.yaml``
3. Fallback default: ``postgresql://astroml:@localhost:5432/astroml``
"""
from __future__ import annotations

import logging
import os
import pathlib
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from astroml.db.pool_health import PoolStats
    from astroml.observability.health import CheckResult

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration with validation."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="astroml", min_length=1, description="Database name")
    user: str = Field(default="astroml", min_length=1, description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout seconds")
    pool_recycle: int = Field(default=1800, description="Pool connection recycle seconds")
    
    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host is not empty."""
        if not v or not v.strip():
            raise ValueError("Database host cannot be empty")
        return v.strip()
    
    def to_url(self) -> str:
        """Convert configuration to PostgreSQL URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatabaseConfig":
        """Create configuration from dictionary with validation."""
        return cls(**data)


def load_database_config(config_path: Optional[pathlib.Path] = None) -> DatabaseConfig:
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
        raise ValueError(
            f"{config_path} is empty. Expected:\n{_database_yaml_template()}"
        )
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
    """Schema-by-example printed in error messages. Mirrors
    config/database.yaml so operators can copy-paste a known-good block."""
    return (
        "database:\n"
        "  host: localhost            # non-empty string\n"
        "  port: 5432                 # 1..65535\n"
        "  name: astroml              # non-empty string\n"
        "  user: astroml              # non-empty string\n"
        "  password: \"\"               # string, may be empty\n"
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
            pool_recycle=config.pool_recycle
        )
    except Exception:
        engine = create_engine(
            resolve_database_url(), 
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800
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
                )
            )
            logger.info("Query profiling enabled in debug mode")
        except ImportError:
            logger.warning("Query profiler module not available")


def get_session() -> Session:
    """Return a new SQLAlchemy session."""
    factory = sessionmaker(bind=get_engine())
    return factory()


def get_pool_stats() -> "PoolStats":
    """Return a snapshot of the shared engine's connection pool (#550)."""
    from astroml.db.pool_health import collect_pool_stats

    return collect_pool_stats(get_engine())


def check_connection_pool() -> "CheckResult":
    """Evaluate connection pool health for the shared engine (#550).

    Returns:
        A ``CheckResult`` named ``"db_pool"`` — ``DEGRADED`` once utilization
        reaches 80% of capacity, ``FAIL`` when the pool is exhausted.
    """
    from astroml.db.pool_health import check_pool

    return check_pool(get_engine())
