"""Schema snapshot regression test for astroml.db.schema (#947).

``astroml/db/schema.py`` is a thin backward-compatible re-export shim
(issue #571) that fans out ``from astroml.db.models import *``. Because it
uses a wildcard import with no ``__all__`` guard on either side, an ORM
model can be silently dropped from the public contract (e.g. renamed,
moved, or accidentally shadowed) without any import error — the shim just
stops re-exporting it.

This test snapshots the *shape* of that contract: which model names,
tables, and columns are reachable via ``astroml.db.schema``. It plays the
same role for this module's re-export surface that an OpenAPI schema
snapshot plays for an HTTP API: pin the current shape so any structural
drift shows up as an explicit, reviewable diff in this file rather than a
silent runtime surprise downstream (e.g. in
``astroml.ingestion.normalizer`` or ``astroml.features.graph.snapshot``,
both of which import ORM models through this exact module).

Deliberately excludes GoldenDataset/GoldenDatasetEntry/Experiment/Variant/
ExperimentResult column-level snapshotting: those models are pre-existing,
partially-implemented stubs (see tests/test_schema.py's now-fixed import,
and its still-pre-existing, separately tracked column-count failures) and
locking in their current incomplete shape here would just create friction
for whoever finishes them. Presence-of-name is still checked for those, so
a full removal is still caught.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase

from astroml.db import schema

# Every ORM model name astroml.db.schema is expected to re-export today.
# Extending this set is fine (and expected) as the schema grows; removing
# or renaming an entry is a breaking change to the re-export contract and
# should be a deliberate, reviewed edit to this snapshot.
EXPECTED_MODEL_NAMES = {
    "Account",
    "Asset",
    "DbModel",
    "Effect",
    "Experiment",
    "ExperimentResult",
    "GoldenDataset",
    "GoldenDatasetEntry",
    "GraphAccount",
    "GraphClaimDetail",
    "GraphEdge",
    "GraphPaymentDetail",
    "GraphTransactionDetail",
    "Ledger",
    "ModelVersion",
    "NormalizedTransaction",
    "Operation",
    "ProcessedLedger",
    "Transaction",
    "Variant",
}

# Stable table-name -> column-name snapshot for the models whose schema is
# considered final/settled (i.e. not one of the known-incomplete stubs
# called out in the module docstring above).
EXPECTED_TABLE_COLUMNS = {
    "ledgers": {
        "sequence",
        "hash",
        "prev_hash",
        "closed_at",
        "successful_transaction_count",
        "failed_transaction_count",
        "operation_count",
        "total_coins",
        "fee_pool",
        "base_fee_in_stroops",
        "protocol_version",
    },
    "transactions": {
        "hash",
        "ledger_sequence",
        "source_account",
        "created_at",
        "fee",
        "operation_count",
        "successful",
        "memo_type",
        "memo",
    },
    "operations": {
        "id",
        "transaction_hash",
        "application_order",
        "type",
        "source_account",
        "destination_account",
        "amount",
        "asset_code",
        "asset_issuer",
        "created_at",
        "details",
    },
    "accounts": None,  # column-level shape not pinned here; name presence only
    "assets": None,
    "graph_accounts": None,
    "graph_edges": None,
    "normalized_transactions": {
        "id",
        "transaction_hash",
        "sender",
        "receiver",
        "asset",
        "amount",
        "timestamp",
    },
}


def _model_by_table_name(table_name: str):
    for attr_name in EXPECTED_MODEL_NAMES:
        model = getattr(schema, attr_name, None)
        if model is not None and getattr(model, "__tablename__", None) == table_name:
            return model
    raise AssertionError(f"No re-exported model maps to table {table_name!r}")


def test_expected_model_names_are_all_reexported():
    """Every name in the snapshot must still be importable from astroml.db.schema."""
    missing = {name for name in EXPECTED_MODEL_NAMES if not hasattr(schema, name)}
    assert not missing, (
        f"astroml.db.schema no longer re-exports: {sorted(missing)}. "
        "If this is intentional, update EXPECTED_MODEL_NAMES in this snapshot test."
    )


def test_reexported_models_are_orm_declarative_classes():
    """Every snapshotted name must actually be a mapped ORM model, not e.g. a helper."""
    for name in EXPECTED_MODEL_NAMES:
        model = getattr(schema, name)
        assert isinstance(model, type) and issubclass(model, DeclarativeBase), (
            f"astroml.db.schema.{name} is no longer a SQLAlchemy declarative model "
            f"(got {model!r})"
        )
        assert hasattr(model, "__tablename__"), f"{name} is missing __tablename__"


def test_settled_table_columns_match_snapshot():
    """Column sets for finalized (non-stub) tables must not silently change."""
    for table_name, expected_columns in EXPECTED_TABLE_COLUMNS.items():
        if expected_columns is None:
            continue
        model = _model_by_table_name(table_name)
        actual_columns = {c.name for c in model.__table__.columns}
        assert actual_columns == expected_columns, (
            f"Column shape for table {table_name!r} drifted.\n"
            f"  Expected: {sorted(expected_columns)}\n"
            f"  Actual:   {sorted(actual_columns)}\n"
            "If this is an intentional schema change, update EXPECTED_TABLE_COLUMNS "
            "in this snapshot test (and add/adjust an Alembic migration)."
        )


def test_no_undocumented_extra_public_names_leak_through():
    """Sanity check: astroml.db.schema shouldn't silently start re-exporting
    unrelated *concrete, mapped* model classes that would widen its contract
    without anyone updating this snapshot.

    Only checks names that look like ORM models (PascalCase, non-dunder) and
    that actually have a ``__tablename__`` (i.e. are mapped to a real table),
    which excludes the abstract ``DeclarativeBase``/``Base`` classes and
    bare aliases such as ``Model = DbModel`` (already covered under its
    canonical name ``DbModel``).
    """
    import astroml.db.models as models_module

    expected_classes = {getattr(schema, name) for name in EXPECTED_MODEL_NAMES}

    public_model_like_names = {
        name
        for name in dir(models_module)
        if not name.startswith("_")
        and name[:1].isupper()
        and isinstance(getattr(models_module, name), type)
        and issubclass(getattr(models_module, name), DeclarativeBase)
        and getattr(models_module, name) is not DeclarativeBase
        and "__tablename__" in vars(getattr(models_module, name))
        and getattr(models_module, name) not in expected_classes
    }

    assert not public_model_like_names, (
        f"astroml.db.models defines new model(s) not covered by this snapshot: "
        f"{sorted(public_model_like_names)}. Add them to EXPECTED_MODEL_NAMES once "
        "their column shape is settled enough to also pin in EXPECTED_TABLE_COLUMNS."
    )
