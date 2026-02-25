"""Database session factory for AstroML.

Resolves the database URL from (in priority order):
1. ``ASTROML_DATABASE_URL`` environment variable
2. ``config/database.yaml``
3. Fallback default: ``postgresql://astroml:@localhost:5432/astroml``
"""
from __future__ import annotations

import os
import pathlib
from functools import lru_cache

import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def resolve_database_url() -> str:
    """Return the database URL, preferring env var over config file."""
    env_url = os.environ.get("ASTROML_DATABASE_URL")
    if env_url:
        return env_url

    config_path = pathlib.Path("config/database.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        db = cfg.get("database", {})
        host = db.get("host", "localhost")
        port = db.get("port", 5432)
        name = db.get("name", "astroml")
        user = db.get("user", "astroml")
        password = db.get("password", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    return "postgresql://astroml:@localhost:5432/astroml"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine."""
    return create_engine(resolve_database_url(), pool_pre_ping=True)


def get_session() -> Session:
    """Return a new SQLAlchemy session."""
    factory = sessionmaker(bind=get_engine())
    return factory()
