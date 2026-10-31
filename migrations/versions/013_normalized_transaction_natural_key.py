"""Add the normalized_transactions natural key.

Revision ID: 013
Revises: 012a
Create Date: 2026-09-25

Revises 012a rather than 012 (issue #832): this revision's own `op.add_column`
calls assumed `normalized_transactions` already existed, but no earlier
migration ever created it — `alembic upgrade head` on a fresh database died
right here with `relation "normalized_transactions" does not exist` until
012a (the same gap already fixed once for `models`/`model_versions` in
010a) was added ahead of it.

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
down_revision: Union[str, None] = "012a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite has no ALTER TABLE ADD CONSTRAINT; batch mode copies the table
    # under the hood to apply the new column/constraint/index together.
    # This is the standard Alembic workaround for that limitation and has
    # no effect on Postgres, which still runs the equivalent plain ALTER
    # TABLE statements.
    with op.batch_alter_table("normalized_transactions") as batch_op:
        batch_op.add_column(sa.Column("ledger_sequence", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("operation_id", sa.BigInteger(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "hop_index",
                sa.SmallInteger(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.create_unique_constraint(
            "uq_normalized_transactions_natural_key",
            ["ledger_sequence", "operation_id", "hop_index"],
        )
        batch_op.create_index(
            "ix_normalized_transactions_operation",
            ["ledger_sequence", "operation_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("normalized_transactions") as batch_op:
        batch_op.drop_index("ix_normalized_transactions_operation")
        batch_op.drop_constraint(
            "uq_normalized_transactions_natural_key",
            type_="unique",
        )
        batch_op.drop_column("hop_index")
        batch_op.drop_column("operation_id")
        batch_op.drop_column("ledger_sequence")
