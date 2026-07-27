"""API backend models — accounts, transactions, fraud alerts, loyalty, model registry.

Revision ID: 004
Revises: 003
Create Date: 2026-06-01

Closes #251 — Database Session & Models
Closes #257 — Model Registry & Versioning
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    # -- api_accounts ----------------------------------------------------------
    op.create_table(
        "api_accounts",
        sa.Column("id", _ID, primary_key=True, autoincrement=True),
        sa.Column("public_key", sa.String(56), nullable=False),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=True),
        sa.Column("balance", sa.Numeric(), nullable=True),
        sa.Column("home_domain", sa.String(253), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("public_key"),
    )
    op.create_index("ix_api_accounts_public_key", "api_accounts", ["public_key"])
    op.create_index("ix_api_accounts_last_active", "api_accounts", ["last_active"])

    # -- api_transactions ------------------------------------------------------
    op.create_table(
        "api_transactions",
        sa.Column("hash", sa.String(64), primary_key=True),
        sa.Column("ledger_sequence", sa.Integer(), nullable=False),
        sa.Column("source_account", sa.String(56), nullable=False),
        sa.Column("destination_account", sa.String(56), nullable=True),
        sa.Column("amount", sa.Numeric(), nullable=True),
        sa.Column("asset_code", sa.String(12), nullable=True),
        sa.Column("asset_issuer", sa.String(56), nullable=True),
        sa.Column("fee", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("operation_type", sa.String(32), nullable=True),
        sa.Column("successful", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("memo_type", sa.String(16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_api_transactions_source_created_at",
                    "api_transactions", ["source_account", "created_at"])
    op.create_index("ix_api_transactions_dest_created_at",
                    "api_transactions", ["destination_account", "created_at"])
    op.create_index("ix_api_transactions_ledger",
                    "api_transactions", ["ledger_sequence"])

    # -- api_fraud_alerts ------------------------------------------------------
    op.create_table(
        "api_fraud_alerts",
        sa.Column("id", _ID, primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.String(56), nullable=False),
        sa.Column("pattern", sa.String(64), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_api_fraud_alerts_account_id", "api_fraud_alerts", ["account_id"])
    op.create_index("ix_api_fraud_alerts_detected_at", "api_fraud_alerts", ["detected_at"])
    op.create_index("ix_api_fraud_alerts_risk_level", "api_fraud_alerts", ["risk_level"])

    # -- loyalty_points --------------------------------------------------------
    op.create_table(
        "loyalty_points",
        sa.Column("id", _ID, primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.String(56), nullable=False),
        sa.Column("balance", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tier", sa.String(32), nullable=False, server_default="bronze"),
        sa.Column("multiplier", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("account_id"),
    )
    op.create_index("ix_loyalty_points_account_id", "loyalty_points", ["account_id"])

    # -- points_transactions ---------------------------------------------------
    op.create_table(
        "points_transactions",
        sa.Column("id", _ID, primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.String(56), nullable=False),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("points", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(128), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_points_transactions_account_id",
                    "points_transactions", ["account_id"])
    op.create_index("ix_points_transactions_created_at",
                    "points_transactions", ["created_at"])

    # -- model_registry --------------------------------------------------------
    op.create_table(
        "model_registry",
        sa.Column("id", _ID, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="inactive"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("name", "version", name="uq_model_registry_name_version"),
    )
    op.create_index("ix_model_registry_name_version",
                    "model_registry", ["name", "version"], unique=True)
    op.create_index("ix_model_registry_status", "model_registry", ["status"])


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("points_transactions")
    op.drop_table("loyalty_points")
    op.drop_table("api_fraud_alerts")
    op.drop_table("api_transactions")
    op.drop_table("api_accounts")
