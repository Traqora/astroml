"""Alembic environment configuration for AstroML.

Resolves the database URL from (in priority order):
1. A ``sqlalchemy.url`` explicitly set on the Alembic ``Config`` object
   (e.g. via ``Config.set_main_option`` before calling ``command.upgrade``/
   ``command.downgrade`` programmatically) — the caller said exactly which
   database to use, so nothing below should override that.
2. ``ASTROML_DATABASE_URL`` environment variable
3. ``config/database.yaml``
4. The ``sqlalchemy.url`` loaded from ``alembic.ini`` on disk

Steps 1 and 4 both read ``sqlalchemy.url`` off the same ``Config`` object;
they're distinguished by comparing against the value freshly parsed from
``alembic.ini`` on disk (step 4's own fallback) — if the ``Config`` holds
something different, a caller must have set it programmatically, so it
takes priority over the environment and the YAML file (issue #857: a test
pointing this ``Config`` at an ephemeral SQLite database for an isolated
up/down cycle was previously always overridden by ``config/database.yaml``,
so it silently exercised the real development Postgres database instead of
the ephemeral one it asked for).
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


def _ini_file_url() -> str:
    """Return ``sqlalchemy.url`` as written in ``alembic.ini`` on disk,
    independent of anything a caller may have set on ``config`` at
    runtime.
    """
    ini_path = config.config_file_name
    if not ini_path or not pathlib.Path(ini_path).exists():
        return ""
    from configparser import ConfigParser

    parser = ConfigParser()
    parser.read(ini_path)
    section = config.config_ini_section
    if parser.has_option(section, "sqlalchemy.url"):
        return parser.get(section, "sqlalchemy.url")
    return ""


def _resolve_url() -> str:
    """Return the database URL, honoring an explicit programmatic override
    first, then falling back through the environment, the YAML config, and
    finally ``alembic.ini``.
    """
    # 1. A caller set sqlalchemy.url on `config` directly (e.g.
    #    `Config.set_main_option(...)` before `command.upgrade`/
    #    `command.downgrade`), which reads back differently from the value
    #    `alembic.ini` would otherwise supply.
    configured_url = config.get_main_option("sqlalchemy.url", "")
    if configured_url and configured_url != _ini_file_url():
        return configured_url

    # 2. Environment variable
    env_url = os.environ.get("ASTROML_DATABASE_URL")
    if env_url:
        return env_url

    # 3. config/database.yaml
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

    # 4. Fall back to alembic.ini
    return configured_url


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
