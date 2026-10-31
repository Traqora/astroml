"""Create the normalized_transactions table.

Revision ID: 012a
Revises: 012
Create Date: 2026-10-19

The ORM model (``astroml.db.models.NormalizedTransaction``) was introduced
without a matching migration, so a database built from ``alembic upgrade
head`` never got this table, and revisions 013 and 014 (which add columns
and an index to it) die with ``no such table: normalized_transactions`` /
``relation "normalized_transactions" does not exist``. This followed the
same pattern as the ``models``/``model_versions`` gap fixed in 010a:
deployments that created the table via ``Base.metadata.create_all`` were
unaffected (creation is skipped when the table already exists), which is
exactly why this had gone unnoticed -- the test suite's migration coverage
(``tests/test_migrations.py``) was, until issue #857's fix to
``migrations/env.py``, silently running against a real Postgres database
instead of the ephemeral one it intended, so it never actually exercised
an ``alembic upgrade head`` against a truly empty database either.

Only the columns and indexes that predate revision 013 are created here;
``ledger_sequence``, ``operation_id``, ``hop_index``, their unique
constraint, and the ``ix_normalized_transactions_operation`` index are
added by 013 (on top of a table that may already hold legacy rows with no
natural key), and the ``ix_normalized_transactions_timestamp_order`` index
is added by 014.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012a"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "normalized_transactions"


def _table_exists() -> bool:
    return _TABLE in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _table_exists():
        return

    # BigInteger on Postgres, Integer on SQLite: matches
    # astroml.db.models.GRAPH_ID_TYPE, since SQLite only auto-increments an
    # INTEGER PRIMARY KEY.
    id_type = sa.BigInteger().with_variant(sa.Integer(), "sqlite")

    op.create_table(
        _TABLE,
        sa.Column("id", id_type, primary_key=True, autoincrement=True),
        sa.Column("transaction_hash", sa.String(length=64), nullable=False),
        sa.Column("sender", sa.String(length=56), nullable=False),
        sa.Column("receiver", sa.String(length=56), nullable=True),
        sa.Column("asset", sa.String(length=70), nullable=False),
        sa.Column("amount", sa.Numeric(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_normalized_transactions_hash",
        _TABLE,
        ["transaction_hash"],
    )
    op.create_index(
        "ix_normalized_transactions_sender_timestamp",
        _TABLE,
        ["sender", "timestamp"],
    )
    op.create_index(
        "ix_normalized_transactions_receiver_timestamp",
        _TABLE,
        ["receiver", "timestamp"],
        postgresql_where=sa.text("receiver IS NOT NULL"),
    )


def downgrade() -> None:
    if not _table_exists():
        return

    op.drop_index(
        "ix_normalized_transactions_receiver_timestamp",
        table_name=_TABLE,
    )
    op.drop_index(
        "ix_normalized_transactions_sender_timestamp",
        table_name=_TABLE,
    )
    op.drop_index("ix_normalized_transactions_hash", table_name=_TABLE)
    op.drop_table(_TABLE)
