"""Index normalized_transactions for time-windowed snapshot reads.

Revision ID: 014
Revises: 013
Create Date: 2026-09-25

Closes #732 — the snapshot builders slice ``normalized_transactions`` by
timestamp range and consume the rows in time order, and ``timestamp`` had no
index at all: every window was a sequential scan plus a sort, and the sort left
rows that share a timestamp (a ledger closes all of its operations in the same
second) in whatever order the plan happened to produce, so two runs over the
same data built different edge sequences.

``ix_normalized_transactions_timestamp_order`` covers the range predicate and
the ordering together — ``(timestamp, ledger_sequence, operation_id,
hop_index)``, the total order the builders sort by. The planner can walk the
index for the range and hand rows back already ordered, so the sort node
disappears along with the non-determinism it introduced.

Because ``timestamp`` leads the index it also serves any other query that
filters or orders on timestamp alone, which is why no separate single-column
index is created.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX = "ix_normalized_transactions_timestamp_order"
_COLUMNS = ["timestamp", "ledger_sequence", "operation_id", "hop_index"]


def upgrade() -> None:
    op.create_index(_INDEX, "normalized_transactions", _COLUMNS)


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="normalized_transactions")
