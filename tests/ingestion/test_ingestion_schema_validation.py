"""Schema validation for ingested records (issue #725).

The parsers subscript raw Horizon dicts directly, so a malformed record raised
``KeyError`` or ``ValueError`` mid-batch, aborted the surrounding work, and
left nothing to report or repair.
"""

from __future__ import annotations

import json

import pytest

from astroml.validation.ingestion_schema import (
    EFFECT_SCHEMA,
    LEDGER_SCHEMA,
    OPERATION_SCHEMA,
    TRANSACTION_SCHEMA,
    FieldSpec,
    RecordSchema,
    iter_valid,
    known_schemas,
    schema_for,
    validate_record,
    validate_records,
)


def valid_operation(**overrides):
    record = {
        "id": 12345,
        "transaction_hash": "a" * 64,
        "type": "payment",
        "source_account": "G" + "A" * 55,
        "created_at": "2024-01-01T00:00:00Z",
        "amount": "10.5",
    }
    record.update(overrides)
    return record


def valid_transaction(**overrides):
    record = {
        "hash": "b" * 64,
        "ledger": 1000,
        "source_account": "G" + "B" * 55,
        "created_at": "2024-01-01T00:00:00Z",
        "fee_charged": 100,
        "operation_count": 1,
        "successful": True,
    }
    record.update(overrides)
    return record


class TestFieldPresence:
    def test_a_complete_record_passes(self):
        assert validate_record(valid_operation(), OPERATION_SCHEMA) == {}

    def test_a_missing_required_field_is_reported_by_name(self):
        reasons = validate_record(valid_operation(transaction_hash=None), OPERATION_SCHEMA)

        assert reasons == {"transaction_hash": "missing required field"}

    def test_an_absent_key_is_the_same_as_a_null(self):
        record = valid_operation()
        del record["source_account"]

        assert validate_record(record, OPERATION_SCHEMA) == {
            "source_account": "missing required field"
        }

    def test_an_absent_optional_field_is_fine(self):
        record = valid_operation()
        del record["amount"]

        assert validate_record(record, OPERATION_SCHEMA) == {}

    def test_every_failure_is_reported_not_just_the_first(self):
        """A record with three problems should be repairable in one pass."""
        reasons = validate_record({"type": "payment"}, OPERATION_SCHEMA)

        assert set(reasons) == {"id", "transaction_hash", "source_account", "created_at"}

    def test_a_non_object_is_rejected_whole(self):
        assert validate_record(["not", "an", "object"], OPERATION_SCHEMA) == {
            "__record__": "expected a JSON object, got list"
        }
        assert "__record__" in validate_record(None, OPERATION_SCHEMA)


class TestFieldTypes:
    def test_an_integer_field_accepts_a_numeric_string(self):
        """Horizon sends numbers as strings; the parsers call ``int()``."""
        assert validate_record(valid_operation(id="12345"), OPERATION_SCHEMA) == {}

    def test_an_integer_field_rejects_a_non_numeric_string(self):
        reasons = validate_record(valid_operation(id="not-a-number"), OPERATION_SCHEMA)

        assert "expected an integer" in reasons["id"]

    def test_an_integer_field_rejects_a_boolean(self):
        """``int(True)`` is 1 — a boolean here is a shape error, not a value."""
        reasons = validate_record(valid_operation(id=True), OPERATION_SCHEMA)

        assert "boolean" in reasons["id"]

    def test_an_integer_field_rejects_a_fractional_float(self):
        reasons = validate_record(valid_operation(id=1.5), OPERATION_SCHEMA)

        assert "expected an integer" in reasons["id"]

    def test_a_string_field_rejects_an_empty_or_blank_value(self):
        assert (
            "non-empty"
            in validate_record(valid_operation(source_account=""), OPERATION_SCHEMA)[
                "source_account"
            ]
        )
        assert (
            "non-empty"
            in validate_record(valid_operation(source_account="   "), OPERATION_SCHEMA)[
                "source_account"
            ]
        )

    def test_a_string_field_rejects_a_number(self):
        reasons = validate_record(valid_operation(type=42), OPERATION_SCHEMA)

        assert "expected a string" in reasons["type"]

    def test_a_boolean_field_accepts_the_string_form(self):
        assert validate_record(valid_transaction(successful="true"), TRANSACTION_SCHEMA) == {}

    def test_a_boolean_field_rejects_an_arbitrary_string(self):
        reasons = validate_record(valid_transaction(successful="maybe"), TRANSACTION_SCHEMA)

        assert "expected a boolean" in reasons["successful"]

    @pytest.mark.parametrize(
        "timestamp",
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00"],
        ids=["zulu", "offset", "naive"],
    )
    def test_a_timestamp_field_accepts_what_the_parser_accepts(self, timestamp):
        """Mirrors ``_parse_datetime``: ``fromisoformat`` after replacing Z."""
        assert validate_record(valid_operation(created_at=timestamp), OPERATION_SCHEMA) == {}

    def test_a_timestamp_field_rejects_a_malformed_value(self):
        reasons = validate_record(valid_operation(created_at="01/02/2024"), OPERATION_SCHEMA)

        assert "ISO-8601" in reasons["created_at"]

    def test_the_validator_agrees_with_the_parser(self):
        """Anything accepted here must not then blow up in the parser."""
        from astroml.ingestion.parsers import _parse_datetime

        for timestamp in ("2024-01-01T00:00:00Z", "2024-06-30T23:59:59+02:00"):
            assert validate_record(valid_operation(created_at=timestamp), OPERATION_SCHEMA) == {}
            _parse_datetime(timestamp)  # must not raise


class TestRangeChecks:
    def test_a_negative_amount_is_rejected(self):
        reasons = validate_record(valid_operation(amount="-5"), OPERATION_SCHEMA)

        assert ">= 0" in reasons["amount"]

    def test_zero_is_allowed(self):
        assert validate_record(valid_operation(amount="0"), OPERATION_SCHEMA) == {}

    def test_a_negative_ledger_sequence_is_rejected(self):
        reasons = validate_record(
            {"sequence": -1, "hash": "x", "closed_at": "2024-01-01T00:00:00Z"}, LEDGER_SCHEMA
        )

        assert ">= 0" in reasons["sequence"]


class TestBatchValidation:
    def test_partitions_a_batch_into_valid_and_rejected(self):
        report = validate_records(
            [valid_operation(), valid_operation(id="bad"), valid_operation()],
            OPERATION_SCHEMA,
            log_rejections=False,
        )

        assert len(report.valid) == 2
        assert report.rejected_count == 1
        assert report.total == 3

    def test_a_rejection_carries_its_index_and_the_record(self):
        """Enough to trace it to a source line and replay it after repair."""
        bad = valid_operation(id="bad")
        report = validate_records([valid_operation(), bad], OPERATION_SCHEMA, log_rejections=False)

        rejection = report.rejected[0]
        assert rejection.index == 1
        assert rejection.record == bad
        assert rejection.schema == "operation"

    def test_a_rejection_identifies_the_record_where_it_can(self):
        report = validate_records(
            [valid_operation(created_at="nope")], OPERATION_SCHEMA, log_rejections=False
        )

        assert report.rejected[0].identifier is not None
        assert "rejected" in report.rejected[0].describe()

    def test_reason_counts_expose_a_systematic_failure(self):
        """A renamed upstream field looks nothing like scattered bad rows."""
        batch = [valid_operation(id=index) for index in range(5)]
        for record in batch:
            del record["created_at"]

        report = validate_records(batch, OPERATION_SCHEMA, log_rejections=False)

        counts = report.reason_counts()
        assert counts["created_at: missing required field"] == 5

    def test_the_rejection_rate_summarises_a_batch(self):
        report = validate_records(
            [valid_operation(), valid_operation(id="bad")],
            OPERATION_SCHEMA,
            log_rejections=False,
        )

        assert report.rejection_rate == pytest.approx(0.5)
        assert report.summary()["valid"] == 1

    def test_an_empty_batch_has_a_zero_rate_rather_than_a_division_error(self):
        report = validate_records([], OPERATION_SCHEMA, log_rejections=False)

        assert report.total == 0
        assert report.rejection_rate == 0.0

    def test_a_rejection_serialises_for_quarantine(self):
        report = validate_records(
            [valid_operation(id="bad")], OPERATION_SCHEMA, log_rejections=False
        )

        payload = report.rejected[0].to_dict()

        assert json.loads(json.dumps(payload))["schema"] == "operation"
        assert payload["record"]["id"] == "bad"


class TestStreamingValidation:
    """Composes with the bounded backfill (#724) without buffering."""

    def test_yields_only_valid_records(self):
        stream = [valid_operation(id=1), valid_operation(id="bad"), valid_operation(id=3)]

        kept = list(iter_valid(stream, OPERATION_SCHEMA, on_reject=lambda _r: None))

        assert [record["id"] for record in kept] == [1, 3]

    def test_reports_rejections_as_they_occur(self):
        rejections = []
        stream = [valid_operation(id="bad"), valid_operation(id=2)]

        list(iter_valid(stream, OPERATION_SCHEMA, on_reject=rejections.append))

        assert len(rejections) == 1
        assert rejections[0].index == 0

    def test_consumes_lazily(self):
        """Nothing is read until it is pulled, so an unbounded stream is safe."""
        pulled = []

        def source():
            for index in range(100):
                pulled.append(index)
                yield valid_operation(id=index)

        generator = iter_valid(source(), OPERATION_SCHEMA, on_reject=lambda _r: None)
        next(generator)

        assert len(pulled) == 1


class TestSchemaCoverage:
    """Each schema must require exactly what its parser dereferences."""

    def test_transaction_schema_covers_the_parser_requirements(self):
        assert validate_record(valid_transaction(), TRANSACTION_SCHEMA) == {}

        for required in ("hash", "ledger", "source_account", "created_at", "fee_charged"):
            record = valid_transaction()
            del record[required]
            assert required in validate_record(record, TRANSACTION_SCHEMA)

    def test_ledger_schema_covers_the_parser_requirements(self):
        record = {"sequence": 5, "hash": "c" * 64, "closed_at": "2024-01-01T00:00:00Z"}

        assert validate_record(record, LEDGER_SCHEMA) == {}

    def test_effect_schema_accepts_a_minimal_effect(self):
        assert validate_record({"id": "0001-1", "type": "account_created"}, EFFECT_SCHEMA) == {}

    def test_an_operation_without_an_amount_is_valid(self):
        """Most operation types carry no amount at all."""
        record = valid_operation(type="set_options")
        del record["amount"]

        assert validate_record(record, OPERATION_SCHEMA) == {}


class TestSchemaLookup:
    def test_finds_each_built_in_schema_by_name(self):
        for name in known_schemas():
            assert schema_for(name).name == name

    def test_an_unknown_name_lists_the_alternatives(self):
        with pytest.raises(KeyError) as excinfo:
            schema_for("nope")

        assert "operation" in str(excinfo.value)

    def test_a_custom_schema_can_be_defined(self):
        schema = RecordSchema(
            name="custom", fields=(FieldSpec("ref", "str"), FieldSpec("count", "int"))
        )

        assert validate_record({"ref": "x", "count": 1}, schema) == {}
        assert "count" in validate_record({"ref": "x"}, schema)

    def test_an_unknown_field_kind_is_rejected_at_definition_time(self):
        with pytest.raises(ValueError, match="unknown field kind"):
            FieldSpec("thing", "uuid")


class TestNoRegressionToParsing:
    """Validation must not change what a good record parses into."""

    def test_a_validated_operation_still_parses(self):
        from astroml.ingestion.parsers import parse_operation

        record = valid_operation()
        assert validate_record(record, OPERATION_SCHEMA) == {}

        parsed = parse_operation(record)
        assert parsed.transaction_hash == record["transaction_hash"]
        assert parsed.type == "payment"

    def test_a_validated_transaction_still_parses(self):
        from astroml.ingestion.parsers import parse_transaction

        record = valid_transaction()
        assert validate_record(record, TRANSACTION_SCHEMA) == {}

        parsed = parse_transaction(record)
        assert parsed.hash == record["hash"]
        assert parsed.ledger_sequence == 1000

    def test_the_records_validation_rejects_are_the_ones_parsing_would_fail_on(self):
        from astroml.ingestion.parsers import parse_transaction

        broken = valid_transaction()
        del broken["hash"]

        assert "hash" in validate_record(broken, TRANSACTION_SCHEMA)
        with pytest.raises(KeyError):
            parse_transaction(broken)
