"""Alembic environment configuration for AstroML.

Resolves the database URL from (in priority order):
1. ``ASTROML_DATABASE_URL`` environment variable
2. ``config/database.yaml``
3. ``sqlalchemy.url`` in ``alembic.ini``
"""
import os
import pathlib
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import our models so Alembic can detect them for --autogenerate.
from astroml.db.schema import Base  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    """Return the database URL, preferring env var over config file."""
    # 1. Environment variable
    env_url = os.environ.get("ASTROML_DATABASE_URL")
    if env_url:
        return env_url

    # 2. config/database.yaml
    config_path = pathlib.Path("config/database.yaml")
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        db = cfg.get("database", {})
        host = db.get("host", "localhost")
        port = db.get("port", 5432)
        name = db.get("name", "astroml")
        user = db.get("user", "astroml")
        password = db.get("password", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    # 3. Fall back to alembic.ini
    return config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL without connecting."""
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connect to the database."""
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _resolve_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
