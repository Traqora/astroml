"""SQLAlchemy ORM models for AstroML storage (issue #571)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ledger(Base):
    __tablename__ = "ledgers"
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    prev_hash: Mapped[Optional[str]] = mapped_column(String(64))
    closed_at: Mapped[datetime] = mapped_column(nullable=False)
    successful_transaction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0")
    failed_transaction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0")
    operation_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_coins: Mapped[Optional[float]] = mapped_column(Numeric)
    fee_pool: Mapped[Optional[float]] = mapped_column(Numeric)
    base_fee_in_stroops: Mapped[Optional[int]] = mapped_column(Integer)
    protocol_version: Mapped[Optional[int]] = mapped_column(Integer)
    transactions: Mapped[list[Transaction]] = relationship(back_populates="ledger")
    __table_args__ = (Index("ix_ledgers_closed_at", "closed_at"),)


class Transaction(Base):
    __tablename__ = "transactions"
    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    ledger_sequence: Mapped[int] = mapped_column(
        Integer, ForeignKey("ledgers.sequence"), nullable=False)
    source_account: Mapped[str] = mapped_column(String(56), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    fee: Mapped[int] = mapped_column(BigInteger, nullable=False)
    operation_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    successful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    memo_type: Mapped[Optional[str]] = mapped_column(String(16))
    memo: Mapped[Optional[str]] = mapped_column(Text)
    ledger: Mapped[Ledger] = relationship(back_populates="transactions")
    operations: Mapped[list[Operation]] = relationship(back_populates="transaction")
    __table_args__ = (
        Index("ix_transactions_source_account_created_at", "source_account", "created_at"),
        Index("ix_transactions_ledger_sequence", "ledger_sequence"),
    )


class Operation(Base):
    __tablename__ = "operations"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("transactions.hash"), nullable=False)
    application_order: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_account: Mapped[str] = mapped_column(String(56), nullable=False)
    destination_account: Mapped[Optional[str]] = mapped_column(String(56))
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    asset_code: Mapped[Optional[str]] = mapped_column(String(12))
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    transaction: Mapped[Transaction] = relationship(back_populates="operations")
    __table_args__ = (
        Index("ix_operations_source_created_at", "source_account", "created_at"),
        Index("ix_operations_dest_created_at", "destination_account",
              "created_at", postgresql_where=(destination_account.isnot(None))),
        Index("ix_operations_transaction_hash", "transaction_hash"),
        Index("ix_operations_type", "type"),
    )


class Account(Base):
    __tablename__ = "accounts"
    account_id: Mapped[str] = mapped_column(String(56), primary_key=True)
    balance: Mapped[Optional[float]] = mapped_column(Numeric)
    sequence: Mapped[Optional[int]] = mapped_column(BigInteger)
    home_domain: Mapped[Optional[str]] = mapped_column(String(32))
    flags: Mapped[int] = mapped_column(Integer, server_default="0")
    last_modified_ledger: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Optional[datetime]] = mapped_column()
    updated_at: Mapped[Optional[datetime]] = mapped_column()
    __table_args__ = (Index("ix_accounts_updated_at", "updated_at"),)


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_type: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_code: Mapped[str] = mapped_column(String(12), nullable=False)
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    first_seen_ledger: Mapped[Optional[int]] = mapped_column(Integer)
    __table_args__ = (Index("ix_assets_code_issuer", "asset_code",
                      func.coalesce(asset_issuer, ""), unique=True),)


GRAPH_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")


class GraphAccount(Base):
    __tablename__ = "graph_accounts"
    id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True, autoincrement=True)
    account_address: Mapped[str] = mapped_column(String(56), nullable=False, unique=True)
    account_type: Mapped[Optional[str]] = mapped_column(String(32))
    first_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now())
    outgoing_edges: Mapped[list[GraphEdge]] = relationship(
        foreign_keys="GraphEdge.source_account_id", back_populates="source_account")
    incoming_edges: Mapped[list[GraphEdge]] = relationship(
        foreign_keys="GraphEdge.destination_account_id", back_populates="destination_account")
    __table_args__ = (Index("ix_graph_accounts_last_seen_at", "last_seen_at"),
                      Index("ix_graph_accounts_account_type", "account_type"),)


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True, autoincrement=True)
    edge_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_account_id: Mapped[int] = mapped_column(
        GRAPH_ID_TYPE, ForeignKey("graph_accounts.id"), nullable=False)
    destination_account_id: Mapped[Optional[int]] = mapped_column(
        GRAPH_ID_TYPE, ForeignKey("graph_accounts.id"))
    asset_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("assets.id"))
    occurred_at: Mapped[datetime] = mapped_column(nullable=False)
    ledger_sequence: Mapped[Optional[int]] = mapped_column(Integer)
    event_index: Mapped[Optional[int]] = mapped_column(Integer)
    transaction_hash: Mapped[Optional[str]] = mapped_column(String(64))
    external_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    status: Mapped[Optional[str]] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    source_account: Mapped[GraphAccount] = relationship(
        foreign_keys=[source_account_id], back_populates="outgoing_edges")
    destination_account: Mapped[Optional[GraphAccount]] = relationship(
        foreign_keys=[destination_account_id], back_populates="incoming_edges")
    asset: Mapped[Optional[Asset]] = relationship()
    transaction_detail: Mapped[Optional[GraphTransactionDetail]] = relationship(
        back_populates="edge", cascade="all, delete-orphan", uselist=False)
    claim_detail: Mapped[Optional[GraphClaimDetail]] = relationship(
        back_populates="edge", cascade="all, delete-orphan", uselist=False)
    payment_detail: Mapped[Optional[GraphPaymentDetail]] = relationship(
        back_populates="edge", cascade="all, delete-orphan", uselist=False)
    __table_args__ = (
        CheckConstraint("edge_type IN ('transaction', 'claim', 'payment')",
                        name="ck_graph_edges_edge_type"),
        CheckConstraint("source_account_id <> destination_account_id OR destination_account_id IS NULL",
                        name="ck_graph_edges_distinct_accounts"),
        UniqueConstraint("edge_type", "external_event_id",
                         name="uq_graph_edges_type_external_event_id"),
        UniqueConstraint("id", "edge_type", name="uq_graph_edges_id_edge_type"),
        Index("ix_graph_edges_occurred_at", "occurred_at"),
        Index("ix_graph_edges_source_occurred_at", "source_account_id", "occurred_at"),
        Index("ix_graph_edges_destination_occurred_at", "destination_account_id", "occurred_at"),
        Index("ix_graph_edges_type_occurred_at", "edge_type", "occurred_at"),
        Index("ix_graph_edges_asset_occurred_at", "asset_id", "occurred_at"),
        Index("ix_graph_edges_status_occurred_at", "status", "occurred_at"),
        Index("ix_graph_edges_tx_hash", "transaction_hash",
              postgresql_where=(transaction_hash.isnot(None))),
        Index("ix_graph_edges_ledger_event", "ledger_sequence", "event_index"),
    )


class GraphTransactionDetail(Base):
    __tablename__ = "graph_transaction_details"
    edge_id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True)
    edge_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="transaction")
    successful: Mapped[Optional[bool]] = mapped_column(Boolean)
    operation_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    fee: Mapped[Optional[int]] = mapped_column(BigInteger)
    memo_type: Mapped[Optional[str]] = mapped_column(String(16))
    memo: Mapped[Optional[str]] = mapped_column(Text)
    details: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    edge: Mapped[GraphEdge] = relationship(back_populates="transaction_detail")
    __table_args__ = (CheckConstraint("edge_type = 'transaction'", name="ck_graph_transaction_details_edge_type"),
                      ForeignKeyConstraint(["edge_id", "edge_type"], ["graph_edges.id", "graph_edges.edge_type"], ondelete="CASCADE"))


class GraphClaimDetail(Base):
    __tablename__ = "graph_claim_details"
    edge_id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True)
    edge_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="claim")
    claim_reference: Mapped[Optional[str]] = mapped_column(String(128))
    claim_status: Mapped[Optional[str]] = mapped_column(String(32))
    expires_at: Mapped[Optional[datetime]] = mapped_column()
    details: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    edge: Mapped[GraphEdge] = relationship(back_populates="claim_detail")
    __table_args__ = (CheckConstraint("edge_type = 'claim'", name="ck_graph_claim_details_edge_type"), ForeignKeyConstraint(["edge_id", "edge_type"], [
                      "graph_edges.id", "graph_edges.edge_type"], ondelete="CASCADE"), Index("ix_graph_claim_details_claim_status", "claim_status"))


class GraphPaymentDetail(Base):
    __tablename__ = "graph_payment_details"
    edge_id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True)
    edge_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="payment")
    payment_reference: Mapped[Optional[str]] = mapped_column(String(128))
    payment_status: Mapped[Optional[str]] = mapped_column(String(32))
    fee_amount: Mapped[Optional[float]] = mapped_column(Numeric)
    settled_at: Mapped[Optional[datetime]] = mapped_column()
    details: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    edge: Mapped[GraphEdge] = relationship(back_populates="payment_detail")
    __table_args__ = (CheckConstraint("edge_type = 'payment'", name="ck_graph_payment_details_edge_type"), CheckConstraint("fee_amount >= 0 OR fee_amount IS NULL", name="ck_graph_payment_details_fee_amount_non_negative"),
                      ForeignKeyConstraint(["edge_id", "edge_type"], ["graph_edges.id", "graph_edges.edge_type"], ondelete="CASCADE"), Index("ix_graph_payment_details_payment_status", "payment_status"))


class Effect(Base):
    __tablename__ = "effects"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account: Mapped[str] = mapped_column(String(56), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    asset_code: Mapped[Optional[str]] = mapped_column(String(12))
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    destination_account: Mapped[Optional[str]] = mapped_column(String(56))
    created_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    __table_args__ = (Index("ix_effects_account_created_at", "account", "created_at"), Index(
        "ix_effects_type_created_at", "type", "created_at"), Index("ix_effects_destination_created_at", "destination_account", "created_at"))


class NormalizedTransaction(Base):
    __tablename__ = "normalized_transactions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sender: Mapped[str] = mapped_column(String(56), nullable=False)
    receiver: Mapped[Optional[str]] = mapped_column(String(56))
    asset: Mapped[str] = mapped_column(String(70), nullable=False)
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    __table_args__ = (Index("ix_normalized_transactions_hash", "transaction_hash"), Index("ix_normalized_transactions_sender_timestamp", "sender",
                      "timestamp"), Index("ix_normalized_transactions_receiver_timestamp", "receiver", "timestamp", postgresql_where=(receiver.isnot(None))))


class DbModel(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    framework: Mapped[str] = mapped_column(String(32), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now())
    versions: Mapped[list[ModelVersion]] = relationship(
        back_populates="model", cascade="all, delete-orphan")
    __table_args__ = (Index("ix_models_framework", "framework"), Index("ix_models_task_type", "task_type"), Index("ix_models_is_active", "is_active"), CheckConstraint(
        "framework IN ('pytorch', 'tensorflow', 'sklearn', 'xgboost', 'lightgbm', 'custom')", name="ck_models_framework"), CheckConstraint("task_type IN ('classification', 'regression', 'anomaly_detection', 'clustering', 'custom')", name="ck_models_task_type"))


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"))
    metrics: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="training")
    # Append-only record of serving transitions (activate / rollback), issue #718.
    # The registry previously wrote these to ``version.metadata``, which is
    # SQLAlchemy's reserved MetaData attribute and not a column — so the lineage
    # it believed it was recording was silently discarded.
    lineage: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"))
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now())
    deployed_at: Mapped[Optional[datetime]] = mapped_column()
    model: Mapped[DbModel] = relationship(back_populates="versions")
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_versions_model_version"), Index("ix_model_versions_model_id", "model_id"), Index("ix_model_versions_status", "status"), Index(
        "ix_model_versions_created_at", "created_at"), CheckConstraint("status IN ('training', 'trained', 'deployed', 'archived', 'failed')", name="ck_models_status"))


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)


class Variant(Base):
    __tablename__ = "variants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)


class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(Integer, nullable=False)


class GoldenDataset(Base):
    __tablename__ = "golden_datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)


class GoldenDatasetEntry(Base):
    __tablename__ = "golden_dataset_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)


class ProcessedLedger(Base):
    __tablename__ = "processed_ledgers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ledger_sequence: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    status: Mapped[Literal["pending", "processing", "completed", "failed"]
                   ] = mapped_column(String(16), nullable=False, server_default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    num_operations: Mapped[Optional[int]] = mapped_column(Integer)
    num_transactions: Mapped[Optional[int]] = mapped_column(Integer)
    __table_args__ = (Index("ix_processed_ledgers_ledger_sequence", "ledger_sequence"), Index(
        "ix_processed_ledgers_status", "status"), Index("ix_processed_ledgers_source", "source"))


# Backward-compatible alias: code that imports `Model` from astroml.db.schema
# still works after the rename to DbModel (issue #571).
Model = DbModel
