"""Alembic schema migration round-trip tests (issue #708).

Every revision must upgrade *and* downgrade cleanly on a pristine database,
so schema drift is caught here instead of in prod. Two layers:

1. Chain integrity (no database): single base, single head, linear
   down_revision chain, and every revision module exposes upgrade +
   downgrade callables.
2. Live round-trip (Postgres): for each revision — upgrade to it, downgrade
   one step, upgrade again — then a full downgrade-to-base and upgrade-to-
   head, asserting key tables exist at the end.

The live tests need a reachable Postgres
(``TEST_POSTGRES_URL``, defaulting to a local ``astroml_alembic_test``
database) because several revisions use PostgreSQL-only DDL
(``JSONB`` columns). They skip — rather than fail — when no database is
reachable, mirroring the repo's DB-backed test convention.
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERSIONS_DIR = REPO_ROOT / "migrations" / "versions"
ALEMBIC_INI = REPO_ROOT / "alembic.ini"

DEFAULT_TEST_URL = "postgresql+psycopg://allison@localhost:5432/postgres"


def _test_db_url() -> str:
    return os.environ.get("TEST_POSTGRES_URL", DEFAULT_TEST_URL)


def _revisions() -> dict[str, dict]:
    """Load every revision module: id -> {down_revision, module}."""
    out = {}
    for path in sorted(VERSIONS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"alembic_rev_{path.stem}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out[module.revision] = {"down_revision": module.down_revision, "module": module}
    assert out, f"no revisions found in {VERSIONS_DIR}"
    return out


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", "migrations")
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


@pytest.fixture()
def pristine_db_url(monkeypatch):
    """A fresh scratch database per test; skipped when Postgres is down.

    ``migrations/env.py`` resolves its URL from ``ASTROML_DATABASE_URL``
    first, so the fixture exports it (scoped to the test via monkeypatch)
    instead of relying on the ini value.
    """
    from sqlalchemy.exc import OperationalError

    base = _test_db_url()
    # Split off the database name so we can create/drop a scratch copy.
    engine = sa.create_engine(base, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect():
            pass
    except OperationalError:
        engine.dispose()
        pytest.skip("Postgres not reachable; set TEST_POSTGRES_URL for migration tests")
        return
    name = f"astroml_migtest_{uuid.uuid4().hex[:12]}"
    with engine.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    engine.dispose()
    url = base.rsplit("/", 1)[0] + f"/{name}"
    monkeypatch.setenv("ASTROML_DATABASE_URL", url)
    yield url
    killer = sa.create_engine(base, isolation_level="AUTOCOMMIT")
    with killer.connect() as conn:
        conn.execute(
            sa.text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = '{name}' AND pid <> pg_backend_pid()"
            )
        )
        conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}"'))
    killer.dispose()


def _current_revision(engine) -> str | None:
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        return ctx.get_current_heads()


class TestAlembicChainIntegrity:
    def test_single_base_and_head(self):
        revs = _revisions()
        bases = [r for r, m in revs.items() if m["down_revision"] is None]
        assert bases, "no base revision (down_revision=None) found"
        assert len(bases) == 1, f"multiple bases (branches): {bases}"
        children: dict[str, list[str]] = {}
        for r, m in revs.items():
            if m["down_revision"] is not None:
                children.setdefault(m["down_revision"], []).append(r)
        heads = [r for r in revs if r not in children]
        assert len(heads) == 1, f"expected one head, found: {heads}"

    def test_linear_chain_no_gaps(self):
        revs = _revisions()
        ordered = [r for r in sorted(revs)]
        assert len(ordered) >= 10, f"expected at least 10 revisions, found {len(ordered)}"
        seen = set()
        current = next(r for r, m in revs.items() if m["down_revision"] is None)
        while current is not None:
            assert current not in seen, f"cycle at {current}"
            seen.add(current)
            nxt = [r for r, m in revs.items() if m["down_revision"] == current]
            assert len(nxt) <= 1, f"branch at {current}: {nxt}"
            current = nxt[0] if nxt else None
        assert seen == set(revs), f"disconnected revisions: {set(revs) - seen}"

    def test_every_revision_has_upgrade_and_downgrade(self):
        for rev, meta in _revisions().items():
            assert callable(getattr(meta["module"], "upgrade", None)), f"{rev} lacks upgrade()"
            assert callable(getattr(meta["module"], "downgrade", None)), f"{rev} lacks downgrade()"


class TestAlembicRoundTrip:
    def test_each_revision_round_trips(self, pristine_db_url):
        """Upgrade → downgrade → upgrade per revision on a pristine DB."""
        from alembic import command

        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            for rev in sorted(_revisions()):
                command.upgrade(cfg, rev)
                assert rev in (_current_revision(engine) or []), f"{rev} not recorded after upgrade"
                command.downgrade(cfg, "-1")
                assert rev not in (
                    _current_revision(engine) or []
                ), f"{rev} still recorded after downgrade"
                command.upgrade(cfg, rev)
                assert rev in (
                    _current_revision(engine) or []
                ), f"{rev} not recorded after re-upgrade"
        finally:
            engine.dispose()

    def test_full_downgrade_to_base_then_head(self, pristine_db_url):
        """Head → base → head proves the whole chain reverses cleanly."""
        from alembic import command

        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            command.upgrade(cfg, "head")
            expected_head = tuple(ScriptDirectory.from_config(cfg).get_heads())
            assert (
                _current_revision(engine) == expected_head
            ), f"head mismatch: {_current_revision(engine)} != {expected_head}"
            command.downgrade(cfg, "base")
            assert _current_revision(engine) in (
                None,
                (),
                [],
            ), "downgrade to base left heads behind"

            command.upgrade(cfg, "head")
            with engine.connect() as conn:
                tables = set(sa.inspect(conn).get_table_names())
            for expected in ("ledgers", "model_registry", "alembic_version"):
                assert expected in tables, f"{expected} missing after upgrade to head"
        finally:
            engine.dispose()


class TestCompositeIndexMigrations:
    """Migration test for composite index (issue #832).

    The chain-integrity and round-trip tests above exercise every
    revision generically, but none of them assert that a *specific*
    composite (multi-column) index actually exists with the right
    columns after upgrading, or that it is really removed on downgrade.
    These tests pin that down for the two composite indexes most central
    to the ingestion/snapshot read paths:
    ``ix_graph_edges_source_occurred_at`` (``graph_edges``, from revision
    002) and ``ix_normalized_transactions_timestamp_order``
    (``normalized_transactions``, from revision 014, issue #732).

    Writing these surfaced a real, separate bug: no revision ever created
    ``normalized_transactions`` at all — revision 013 assumed it already
    existed and died with ``relation "normalized_transactions" does not
    exist`` on a fresh database, the same gap already fixed once for
    ``models``/``model_versions`` in revision 010a. Fixed by adding
    revision 012a ahead of 013.
    """

    def test_graph_edges_composite_index_exists_with_correct_columns(self, pristine_db_url):
        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            from alembic import command

            command.upgrade(cfg, "head")
            indexes = sa.inspect(engine).get_indexes("graph_edges")
            composite = next(
                (i for i in indexes if i["name"] == "ix_graph_edges_source_occurred_at"), None
            )
            assert composite is not None, "ix_graph_edges_source_occurred_at missing after upgrade"
            assert composite["column_names"] == ["source_account_id", "occurred_at"]
        finally:
            engine.dispose()

    def test_normalized_transactions_composite_index_exists_with_correct_columns(
        self, pristine_db_url
    ):
        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            from alembic import command

            command.upgrade(cfg, "head")
            indexes = sa.inspect(engine).get_indexes("normalized_transactions")
            composite = next(
                (i for i in indexes if i["name"] == "ix_normalized_transactions_timestamp_order"),
                None,
            )
            assert (
                composite is not None
            ), "ix_normalized_transactions_timestamp_order missing after upgrade"
            assert composite["column_names"] == [
                "timestamp",
                "ledger_sequence",
                "operation_id",
                "hop_index",
            ]
        finally:
            engine.dispose()

    def test_normalized_transactions_table_itself_exists_after_upgrade(self, pristine_db_url):
        """Regression guard for the missing-table bug found while writing
        this test class: revisions 013/014 both assume
        ``normalized_transactions`` already exists, so this asserts the
        table is actually present, not just its composite index."""
        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            from alembic import command

            command.upgrade(cfg, "head")
            tables = sa.inspect(engine).get_table_names()
            assert "normalized_transactions" in tables
        finally:
            engine.dispose()

    def test_normalized_transactions_composite_index_is_removed_on_downgrade(self, pristine_db_url):
        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            from alembic import command

            command.upgrade(cfg, "head")
            command.downgrade(cfg, "012a")
            tables = sa.inspect(engine).get_table_names()
            # The composite index's own revision (014) and the natural-key
            # revision (013) are both downgraded past; only the table
            # itself (created by 012a) should remain.
            assert "normalized_transactions" in tables
            indexes = sa.inspect(engine).get_indexes("normalized_transactions")
            assert not any(
                i["name"] == "ix_normalized_transactions_timestamp_order" for i in indexes
            )
        finally:
            engine.dispose()

    def test_full_downgrade_removes_normalized_transactions_entirely(self, pristine_db_url):
        cfg = _alembic_config(pristine_db_url)
        engine = sa.create_engine(pristine_db_url)
        try:
            from alembic import command

            command.upgrade(cfg, "head")
            command.downgrade(cfg, "base")
            tables = sa.inspect(engine).get_table_names()
            assert "normalized_transactions" not in tables

            command.upgrade(cfg, "head")
            tables_after_reupgrade = sa.inspect(engine).get_table_names()
            assert "normalized_transactions" in tables_after_reupgrade
        finally:
            engine.dispose()
