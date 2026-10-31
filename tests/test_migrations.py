"""Tests for Alembic database migrations.

Verifies that migrations are reversible and that the schema matches
the declared ORM models after each upgrade/downgrade cycle.

Resolves #515.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import sqlalchemy as sa
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

    def test_upgrade_actually_targets_the_ephemeral_database(
        self, alembic_config: Config, tmp_path
    ):
        """The upgrade runs against the Config's own sqlalchemy.url, not
        whatever ``config/database.yaml`` or ``ASTROML_DATABASE_URL``
        happens to resolve to (issue #857).

        ``config/database.yaml`` is checked into this repository pointing
        at a real Postgres database, and used to take priority over an
        explicit ``Config.set_main_option("sqlalchemy.url", ...)`` call in
        ``migrations/env.py``'s URL resolution, so this fixture's temporary
        SQLite file was silently ignored in favor of that real database.
        This asserts the ephemeral file this test actually asked for was
        the one written to.
        """
        db_path = tmp_path / "test.db"
        assert not db_path.exists()

        with patch.dict(os.environ, {"ASTROML_DATABASE_URL": ""}, clear=False):
            command.upgrade(alembic_config, "head")

        assert db_path.exists(), (
            "alembic upgrade did not write to the ephemeral SQLite file "
            "the test's own Config pointed it at"
        )

    def test_normalized_transactions_table_created_by_migrations(self, alembic_config: Config):
        """``normalized_transactions`` (added by 012a) exists at head, with
        the natural-key columns 013 adds on top of it.

        This table has no corresponding ``op.create_table`` anywhere except
        012a: the ORM model existed without one (the same class of gap
        010a already fixed for ``models``/``model_versions``), so revisions
        013 and 014 died with "no such table" the moment migrations
        actually ran against a fresh database rather than the real,
        already-populated Postgres database issue #857 was silently
        testing against.
        """
        with patch.dict(os.environ, {"ASTROML_DATABASE_URL": ""}, clear=False):
            command.upgrade(alembic_config, "head")

        db_url = alembic_config.get_main_option("sqlalchemy.url")
        assert db_url is not None
        engine = sa.create_engine(db_url)
        try:
            inspector = sa.inspect(engine)
            assert "normalized_transactions" in inspector.get_table_names()
            columns = {c["name"] for c in inspector.get_columns("normalized_transactions")}
            assert {
                "id",
                "transaction_hash",
                "sender",
                "receiver",
                "asset",
                "amount",
                "timestamp",
                "ledger_sequence",
                "operation_id",
                "hop_index",
            }.issubset(columns)
        finally:
            engine.dispose()
