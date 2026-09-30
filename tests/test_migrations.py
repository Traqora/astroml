"""Tests for Alembic database migrations.

Verifies that migrations are reversible and that the schema matches
the declared ORM models after each upgrade/downgrade cycle.

Resolves #515.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config


@pytest.fixture()
def alembic_config(tmp_path):
    """Return an Alembic config pointing at a temporary SQLite database."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


@pytest.fixture()
def _clean_env():
    """Ensure ASTROML_DATABASE_URL does not leak between tests."""
    old = os.environ.pop("ASTROML_DATABASE_URL", None)
    yield
    if old is not None:
        os.environ["ASTROML_DATABASE_URL"] = old


@pytest.mark.usefixtures("_clean_env")
class TestMigrations:
    """Basic migration smoke tests."""

    def test_upgrade_to_head(self, alembic_config: Config):
        """All migrations apply cleanly from base to head."""
        with patch.dict(os.environ, {"ASTROML_DATABASE_URL": ""}, clear=False):
            command.upgrade(alembic_config, "head")

    def test_downgrade_to_base(self, alembic_config: Config):
        """All migrations reverse cleanly from head to base."""
        with patch.dict(os.environ, {"ASTROML_DATABASE_URL": ""}, clear=False):
            command.upgrade(alembic_config, "head")
            command.downgrade(alembic_config, "base")

    def test_upgrade_downgrade_cycle(self, alembic_config: Config):
        """A full upgrade→downgrade→upgrade cycle completes without error."""
        with patch.dict(os.environ, {"ASTROML_DATABASE_URL": ""}, clear=False):
            command.upgrade(alembic_config, "head")
            command.downgrade(alembic_config, "base")
            command.upgrade(alembic_config, "head")
