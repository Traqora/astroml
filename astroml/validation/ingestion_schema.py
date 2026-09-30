"""Schema validation for ingested Horizon records (issue #725).

The parsers in :mod:`astroml.ingestion.parsers` read raw Horizon dicts by
subscripting them — ``data["hash"]``, ``int(data["ledger"])``,
``_parse_datetime(data["created_at"])``. A record missing a field raises
``KeyError`` mid-batch; one with a malformed field raises ``ValueError`` from
inside ``int()`` or ``datetime`` parsing. Either way the exception names the
field at best and nothing at worst, it aborts the surrounding batch, and the
offending record is gone — there is no way to report what was rejected or to
repair and replay it.

This module validates a record *before* it reaches a parser and, instead of
raising, returns the reasons. A caller then persists what is valid and keeps
the rejects with enough detail to fix them:

    report = validate_records(raw_operations, OPERATION_SCHEMA)
    persist(report.valid)
    write_quarantine(report.rejected)

Validation is deliberately structural — presence, type, coercibility, obvious
range errors. It does not check that a hash exists on-chain or that an amount
is economically sensible; those are questions for a later stage that has the
ledger to compare against.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("astroml.validation.ingestion_schema")

__all__ = [
    "EFFECT_SCHEMA",
    "FieldSpec",
    "LEDGER_SCHEMA",
    "OPERATION_SCHEMA",
    "RecordSchema",
    "RejectedRecord",
    "TRANSACTION_SCHEMA",
    "ValidationReport",
    "iter_valid",
    "validate_record",
    "validate_records",
]


# ---------------------------------------------------------------------------
# Field specifications
# ---------------------------------------------------------------------------


def _check_int(value: Any) -> str | None:
    """Accept anything ``int()`` accepts losslessly."""
    if isinstance(value, bool):
        # `int(True)` is 1. A boolean in a numeric field is a shape error in
        # the source, not a value to silently coerce.
        return "expected an integer, got a boolean"
    if isinstance(value, int):
        return None
    if isinstance(value, str):
        try:
            int(value)
        except ValueError:
            return f"expected an integer, got {value!r}"
        return None
    if isinstance(value, float):
        return None if value.is_integer() else f"expected an integer, got {value!r}"
    return f"expected an integer, got {type(value).__name__}"


def _check_float(value: Any) -> str | None:
    if isinstance(value, bool):
        return "expected a number, got a boolean"
    if isinstance(value, (int, float)):
        return None
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return f"expected a number, got {value!r}"
        return None
    return f"expected a number, got {type(value).__name__}"


def _check_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return f"expected a string, got {type(value).__name__}"
    if not value.strip():
        return "expected a non-empty string"
    return None


def _check_bool(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return None
    return f"expected a boolean, got {type(value).__name__}"


def _check_datetime(value: Any) -> str | None:
    """Accept what :func:`astroml.ingestion.parsers._parse_datetime` accepts.

    That function does ``datetime.fromisoformat(iso.replace("Z", "+00:00"))``,
    so this mirrors it exactly rather than approximating — a record this
    accepts and the parser then rejects would defeat the point.
    """
    if isinstance(value, datetime):
        return None
    if not isinstance(value, str):
        return f"expected an ISO-8601 timestamp, got {type(value).__name__}"
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return f"expected an ISO-8601 timestamp, got {value!r}"
    return None


_CHECKERS: dict[str, Callable[[Any], str | None]] = {
    "int": _check_int,
    "float": _check_float,
    "str": _check_str,
    "bool": _check_bool,
    "datetime": _check_datetime,
}


@dataclass(frozen=True)
class FieldSpec:
    """One field's expectations.

    Attributes:
        name: Key in the raw record.
        kind: One of ``int``, ``float``, ``str``, ``bool``, ``datetime``.
        required: Whether the field must be present and non-null.
        min_value: Optional inclusive lower bound for numeric fields.
    """

    name: str
    kind: str = "str"
    required: bool = True
    min_value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _CHECKERS:
            raise ValueError(
                f"unknown field kind {self.kind!r}; expected one of {sorted(_CHECKERS)}"
            )

    def check(self, record: dict) -> str | None:
        """Return a human-readable reason, or ``None`` when the field is fine."""
        if self.name not in record or record[self.name] is None:
            # An optional field that is absent is not a problem; a required one
            # is, and saying which is missing is the whole point of reporting.
            return None if not self.required else "missing required field"

        value = record[self.name]
        reason = _CHECKERS[self.kind](value)
        if reason is not None:
            return reason

        if self.min_value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if numeric < self.min_value:
                return f"expected a value >= {self.min_value:g}, got {value!r}"

        return None


@dataclass(frozen=True)
class RecordSchema:
    """A named set of field specifications."""

    name: str
    fields: tuple[FieldSpec, ...]

    def reasons(self, record: Any) -> dict[str, str]:
        """Every field failure in ``record``, keyed by field name.

        All fields are checked rather than stopping at the first: a record
        with three problems should be repairable in one pass, not three.
        """
        if not isinstance(record, dict):
            return {"__record__": f"expected a JSON object, got {type(record).__name__}"}

        failures: dict[str, str] = {}
        for spec in self.fields:
            reason = spec.check(record)
            if reason is not None:
                failures[spec.name] = reason
        return failures


# ---------------------------------------------------------------------------
# Schemas, mirroring what the parsers actually dereference
# ---------------------------------------------------------------------------

LEDGER_SCHEMA = RecordSchema(
    name="ledger",
    fields=(
        FieldSpec("sequence", "int", min_value=0),
        FieldSpec("hash", "str"),
        FieldSpec("closed_at", "datetime"),
        FieldSpec("prev_hash", "str", required=False),
        FieldSpec("successful_transaction_count", "int", required=False, min_value=0),
        FieldSpec("failed_transaction_count", "int", required=False, min_value=0),
        FieldSpec("operation_count", "int", required=False, min_value=0),
        FieldSpec("total_coins", "float", required=False),
        FieldSpec("fee_pool", "float", required=False),
        FieldSpec("base_fee_in_stroops", "int", required=False, min_value=0),
        FieldSpec("protocol_version", "int", required=False, min_value=0),
    ),
)

TRANSACTION_SCHEMA = RecordSchema(
    name="transaction",
    fields=(
        FieldSpec("hash", "str"),
        FieldSpec("ledger", "int", min_value=0),
        FieldSpec("source_account", "str"),
        FieldSpec("created_at", "datetime"),
        FieldSpec("fee_charged", "int", min_value=0),
        FieldSpec("operation_count", "int", min_value=0),
        FieldSpec("successful", "bool"),
        FieldSpec("memo_type", "str", required=False),
        FieldSpec("memo", "str", required=False),
    ),
)

OPERATION_SCHEMA = RecordSchema(
    name="operation",
    fields=(
        FieldSpec("id", "int", min_value=0),
        FieldSpec("transaction_hash", "str"),
        FieldSpec("type", "str"),
        FieldSpec("source_account", "str"),
        FieldSpec("created_at", "datetime"),
        # Not required: only payment-like operations carry an amount, and
        # rejecting a `set_options` for lacking one would quarantine most of
        # the ledger.
        FieldSpec("amount", "float", required=False, min_value=0),
    ),
)

EFFECT_SCHEMA = RecordSchema(
    name="effect",
    fields=(
        FieldSpec("id", "str"),
        FieldSpec("type", "str"),
        FieldSpec("account", "str", required=False),
        FieldSpec("created_at", "datetime", required=False),
    ),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RejectedRecord:
    """A record that failed validation, with enough context to repair it.

    Attributes:
        index: Position in the input sequence, so a rejection can be traced
            back to a line in the source file.
        schema: Name of the schema it was checked against.
        reasons: Field name to failure reason.
        record: The record itself, retained so it can be replayed after repair.
        identifier: Best-effort id for logs — the record's ``hash``, ``id`` or
            ``sequence`` when it has a usable one.
    """

    index: int
    schema: str
    reasons: dict[str, str]
    record: Any
    identifier: str | None = None

    def describe(self) -> str:
        """One-line summary for a log or a report."""
        subject = self.identifier or f"index {self.index}"
        detail = "; ".join(f"{field}: {reason}" for field, reason in sorted(self.reasons.items()))
        return f"{self.schema} {subject} rejected ({detail})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form for a quarantine file."""
        return {
            "index": self.index,
            "schema": self.schema,
            "identifier": self.identifier,
            "reasons": dict(self.reasons),
            "record": self.record,
        }


@dataclass
class ValidationReport:
    """Outcome of validating a batch."""

    schema: str
    valid: list[Any] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.valid) + len(self.rejected)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def rejection_rate(self) -> float:
        """Fraction rejected; ``0.0`` for an empty batch."""
        return self.rejected_count / self.total if self.total else 0.0

    def reason_counts(self) -> dict[str, int]:
        """How often each ``field: reason`` occurred.

        A batch rejected for one systematic cause — a renamed Horizon field,
        say — looks completely different from one with scattered bad rows, and
        this is what makes that visible without reading every rejection.
        """
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            for field_name, reason in rejection.reasons.items():
                key = f"{field_name}: {reason}"
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def summary(self) -> dict[str, Any]:
        """Compact summary suitable for logging or a metrics label."""
        return {
            "schema": self.schema,
            "total": self.total,
            "valid": len(self.valid),
            "rejected": self.rejected_count,
            "rejection_rate": round(self.rejection_rate, 6),
            "reasons": self.reason_counts(),
        }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _identify(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("hash", "transaction_hash", "id", "sequence"):
        value = record.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return f"{key}={value}"
    return None


def validate_record(record: Any, schema: RecordSchema) -> dict[str, str]:
    """Validate one record. Returns an empty dict when it is valid."""
    return schema.reasons(record)


def validate_records(
    records: Iterable[Any],
    schema: RecordSchema,
    *,
    log_rejections: bool = True,
) -> ValidationReport:
    """Validate a batch, partitioning it into valid records and rejections.

    Note this materialises the batch — it is meant for a bounded chunk, which
    is how ingestion already writes (see ``BatchBuffer``). For an unbounded
    stream use :func:`iter_valid`, which keeps only one record at a time.
    """
    report = ValidationReport(schema=schema.name)

    for index, record in enumerate(records):
        reasons = schema.reasons(record)
        if reasons:
            rejection = RejectedRecord(
                index=index,
                schema=schema.name,
                reasons=reasons,
                record=record,
                identifier=_identify(record),
            )
            report.rejected.append(rejection)
            if log_rejections:
                logger.warning("%s", rejection.describe())
        else:
            report.valid.append(record)

    if report.rejected:
        logger.info(
            "Schema validation for %s: %d/%d rejected",
            schema.name,
            report.rejected_count,
            report.total,
        )

    return report


def iter_valid(
    records: Iterable[Any],
    schema: RecordSchema,
    *,
    on_reject: Callable[[RejectedRecord], None] | None = None,
) -> Iterator[Any]:
    """Yield only the valid records from a stream.

    The streaming counterpart to :func:`validate_records`: nothing is
    accumulated, so this composes with a large backfill (#724) without
    reintroducing a per-range buffer. Rejections go to ``on_reject`` as they
    occur — collect them, count them, or write them straight to quarantine.
    """
    for index, record in enumerate(records):
        reasons = schema.reasons(record)
        if not reasons:
            yield record
            continue

        rejection = RejectedRecord(
            index=index,
            schema=schema.name,
            reasons=reasons,
            record=record,
            identifier=_identify(record),
        )
        if on_reject is not None:
            on_reject(rejection)
        else:
            logger.warning("%s", rejection.describe())


def schema_for(name: str) -> RecordSchema:
    """Look a schema up by record type name."""
    schemas: dict[str, RecordSchema] = {
        LEDGER_SCHEMA.name: LEDGER_SCHEMA,
        TRANSACTION_SCHEMA.name: TRANSACTION_SCHEMA,
        OPERATION_SCHEMA.name: OPERATION_SCHEMA,
        EFFECT_SCHEMA.name: EFFECT_SCHEMA,
    }
    try:
        return schemas[name]
    except KeyError:
        raise KeyError(
            f"unknown record schema {name!r}; available: {', '.join(sorted(schemas))}"
        ) from None


def known_schemas() -> Sequence[str]:
    """Names of the built-in schemas."""
    return (LEDGER_SCHEMA.name, TRANSACTION_SCHEMA.name, OPERATION_SCHEMA.name, EFFECT_SCHEMA.name)
