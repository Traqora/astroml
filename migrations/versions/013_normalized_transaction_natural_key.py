"""Add the normalized_transactions natural key.

Revision ID: 013
Revises: 012
Create Date: 2026-09-25

Closes #728 — writes to ``normalized_transactions`` were not idempotent on
retry: the table's only key was the surrogate ``id``, which a freshly
normalized row does not have, so ``session.merge()`` inserted a duplicate on
every replayed Horizon event, resumed batch, or refetched page.

This adds the natural key the ingestion path writes against,
``(ledger_sequence, operation_id, hop_index)``:

* ``ledger_sequence`` is recovered from the operation's toid -- Horizon packs
  ``ledger << 32 | tx_order << 12 | op_index`` into it -- so no extra payload
  field is needed.
* ``hop_index`` separates the rows a single path-payment operation decomposes
  into, which were previously told apart by a ``_hopN`` suffix appended to
  ``transaction_hash``.  That suffix pushed the value past the column's 64
  characters, so a path payment could not be stored on Postgres at all.

All three columns are nullable/`0`-defaulted rather than ``NOT NULL`` so this
applies to a table that may already hold rows.  Rows written before this
migration have no operation id to backfill from, so they keep NULLs; Postgres
and SQLite both treat NULLs as distinct under a UNIQUE constraint, which
leaves those legacy rows alone while enforcing the key for everything written
after this point.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "normalized_transactions",
        sa.Column("ledger_sequence", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "normalized_transactions",
        sa.Column("operation_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "normalized_transactions",
        sa.Column(
            "hop_index",
            sa.SmallInteger(),
            nullable=False,
            server_default="0",
        ),
    )
    op.create_unique_constraint(
        "uq_normalized_transactions_natural_key",
        "normalized_transactions",
        ["ledger_sequence", "operation_id", "hop_index"],
    )
    op.create_index(
        "ix_normalized_transactions_operation",
        "normalized_transactions",
        ["ledger_sequence", "operation_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_normalized_transactions_operation",
        table_name="normalized_transactions",
    )
    op.drop_constraint(
        "uq_normalized_transactions_natural_key",
        "normalized_transactions",
        type_="unique",
    )
    op.drop_column("normalized_transactions", "hop_index")
    op.drop_column("normalized_transactions", "operation_id")
    op.drop_column("normalized_transactions", "ledger_sequence")
