"""Tests for the public v1 exception types in ``astroml.utils.exceptions`` (#1234).

Covers the hierarchy callers catch on, the stable machine-readable codes, and the
boundary between internal detail (message, context, cause) and what is safe to show
to an API client (``to_public_dict``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astroml.utils import exceptions
from astroml.utils.exceptions import (
    AstroMLError,
    ConfigurationError,
    DatabaseError,
    FeatureError,
    IngestionError,
    ModelError,
    ParseError,
    SourceUnavailableError,
)

PUBLIC_ERRORS = [
    AstroMLError,
    ConfigurationError,
    IngestionError,
    ParseError,
    SourceUnavailableError,
    FeatureError,
    ModelError,
    DatabaseError,
]

EXPECTED_CODES = {
    AstroMLError: "ASTROML_ERROR",
    ConfigurationError: "CONFIGURATION_ERROR",
    IngestionError: "INGESTION_ERROR",
    ParseError: "PARSE_ERROR",
    SourceUnavailableError: "SOURCE_UNAVAILABLE",
    FeatureError: "FEATURE_ERROR",
    ModelError: "MODEL_ERROR",
    DatabaseError: "DATABASE_ERROR",
}

ERRORS_DOC = Path(__file__).resolve().parents[1] / "docs" / "v1-errors.md"


class TestBackwardsCompatibility:
    """Existing callers use ``AstroMLError(message, **context)`` and the original four stages."""

    def test_message_and_context_are_kept(self) -> None:
        err = AstroMLError("boom", run_id="r1", attempt=2)

        assert str(err) == "boom"
        assert err.message == "boom"
        assert err.context == {"run_id": "r1", "attempt": 2}

    def test_context_defaults_to_empty(self) -> None:
        assert AstroMLError("boom").context == {}

    @pytest.mark.parametrize("cls", [IngestionError, FeatureError, ModelError, DatabaseError])
    def test_original_stage_errors_construct_the_same_way(self, cls: type[AstroMLError]) -> None:
        err = cls("failed", table="transactions")

        assert isinstance(err, AstroMLError)
        assert str(err) == "failed"
        assert err.context == {"table": "transactions"}


class TestHierarchy:
    @pytest.mark.parametrize("cls", PUBLIC_ERRORS)
    def test_every_public_error_is_an_astroml_error(self, cls: type[AstroMLError]) -> None:
        assert issubclass(cls, AstroMLError)
        assert issubclass(cls, Exception)

    def test_ingestion_has_parse_and_source_unavailable_children(self) -> None:
        assert issubclass(ParseError, IngestionError)
        assert issubclass(SourceUnavailableError, IngestionError)

    def test_stages_do_not_catch_each_other(self) -> None:
        for sibling in (ConfigurationError, FeatureError, ModelError, DatabaseError):
            assert not issubclass(sibling, IngestionError)
        assert not issubclass(ParseError, SourceUnavailableError)
        assert not issubclass(SourceUnavailableError, ParseError)

    def test_catching_the_base_class_handles_any_stage(self) -> None:
        for cls in PUBLIC_ERRORS:
            with pytest.raises(AstroMLError):
                raise cls("x")

    def test_catching_ingestion_handles_its_children_only(self) -> None:
        with pytest.raises(IngestionError):
            raise ParseError("x")
        with pytest.raises(IngestionError):
            raise SourceUnavailableError("x")
        with pytest.raises(DatabaseError):
            try:
                raise DatabaseError("x")
            except IngestionError:  # pragma: no cover - must not be reached
                pytest.fail("a DatabaseError must not be caught as an IngestionError")

    def test_all_exports_exactly_the_public_errors(self) -> None:
        assert set(exceptions.__all__) == {cls.__name__ for cls in PUBLIC_ERRORS}
        for name in exceptions.__all__:
            assert hasattr(exceptions, name)


class TestStableCodes:
    @pytest.mark.parametrize("cls,code", list(EXPECTED_CODES.items()))
    def test_codes_are_stable(self, cls: type[AstroMLError], code: str) -> None:
        assert cls.code == code
        assert cls("x").code == code

    def test_codes_are_unique(self) -> None:
        codes = [cls.code for cls in PUBLIC_ERRORS]
        assert len(codes) == len(set(codes))

    @pytest.mark.parametrize("cls", PUBLIC_ERRORS)
    def test_every_error_has_a_public_message_and_docstring(self, cls: type[AstroMLError]) -> None:
        assert cls.public_message.strip()
        assert cls.__doc__ and cls.__doc__.strip()

    def test_only_source_unavailable_is_retryable(self) -> None:
        assert {cls for cls in PUBLIC_ERRORS if cls.retryable} == {SourceUnavailableError}
        assert SourceUnavailableError("x").retryable is True
        assert DatabaseError("x").retryable is False


class TestPublicRepresentation:
    def test_contains_only_code_message_and_retryable(self) -> None:
        public = ModelError("internal detail").to_public_dict()

        assert public == {
            "code": "MODEL_ERROR",
            "message": "The model could not complete the request.",
            "retryable": False,
        }

    def test_never_leaks_the_internal_message_or_context(self) -> None:
        err = DatabaseError(
            "SELECT * FROM users WHERE token='hunter2' failed on db-prod-1:5432",
            host="db-prod-1",
            password="hunter2",
        )

        rendered = json.dumps(err.to_public_dict())

        assert "hunter2" not in rendered
        assert "db-prod-1" not in rendered
        assert "SELECT" not in rendered

    def test_never_leaks_the_cause_or_a_traceback(self) -> None:
        try:
            try:
                raise ConnectionError("connect to 10.0.0.7 refused")
            except ConnectionError as exc:
                raise SourceUnavailableError("horizon down") from exc
        except SourceUnavailableError as err:
            rendered = json.dumps(err.to_public_dict())

        assert "10.0.0.7" not in rendered
        assert "Traceback" not in rendered
        assert "ConnectionError" not in rendered

    @pytest.mark.parametrize("cls", PUBLIC_ERRORS)
    def test_is_json_serialisable(self, cls: type[AstroMLError]) -> None:
        assert json.loads(json.dumps(cls("x").to_public_dict()))["code"] == cls.code

    def test_public_dict_is_a_fresh_copy(self) -> None:
        err = FeatureError("x")
        err.to_public_dict()["message"] = "tampered"

        assert err.to_public_dict()["message"] == FeatureError.public_message


class TestCausePreservation:
    def test_raise_from_keeps_the_original_cause(self) -> None:
        original = KeyError("closed_at")

        with pytest.raises(ParseError) as caught:
            try:
                raise original
            except KeyError as exc:
                raise ParseError("ledger payload is missing a field") from exc

        assert caught.value.__cause__ is original

    def test_context_does_not_replace_the_cause(self) -> None:
        original = ValueError("bad int")

        try:
            try:
                raise original
            except ValueError as exc:
                raise IngestionError("could not read value", source="horizon") from exc
        except IngestionError as err:
            assert err.__cause__ is original
            assert err.context == {"source": "horizon"}


class TestParseError:
    def test_message_names_the_ledger_and_transaction(self) -> None:
        err = ParseError("bad timestamp", ledger_sequence=12554, transaction_hash="abc123")

        assert str(err) == "bad timestamp [ledger=12554 tx=abc123]"
        assert err.message == "bad timestamp [ledger=12554 tx=abc123]"

    def test_identifiers_are_attributes_and_in_context(self) -> None:
        err = ParseError(
            "bad timestamp", ledger_sequence=12554, transaction_hash="abc123", field="created_at"
        )

        assert err.ledger_sequence == 12554
        assert err.transaction_hash == "abc123"
        assert err.field == "created_at"
        assert err.context == {
            "ledger_sequence": 12554,
            "transaction_hash": "abc123",
            "field": "created_at",
        }
        assert "field=created_at" in str(err)

    def test_extra_context_is_kept_alongside_the_identifiers(self) -> None:
        err = ParseError("bad", ledger_sequence=7, source="stream")

        assert err.context == {"source": "stream", "ledger_sequence": 7}

    def test_only_known_identifiers_appear(self) -> None:
        only_ledger = ParseError("bad", ledger_sequence=9)
        only_tx = ParseError("bad", transaction_hash="def456")

        assert str(only_ledger) == "bad [ledger=9]"
        assert only_ledger.transaction_hash is None
        assert str(only_tx) == "bad [tx=def456]"
        assert only_tx.ledger_sequence is None

    def test_zero_is_a_real_ledger_sequence_not_missing(self) -> None:
        assert str(ParseError("bad", ledger_sequence=0)) == "bad [ledger=0]"

    def test_without_identifiers_the_message_is_unchanged(self) -> None:
        err = ParseError("bad payload")

        assert str(err) == "bad payload"
        assert err.context == {}
        assert err.field is None

    def test_public_dict_carries_the_on_chain_identifiers_only(self) -> None:
        err = ParseError(
            "internal: row 5 of batch 9 failed",
            ledger_sequence=12554,
            transaction_hash="abc123",
            field="created_at",
            batch="batch-9",
        )

        public = err.to_public_dict()

        assert public == {
            "code": "PARSE_ERROR",
            "message": "Ledger data could not be parsed.",
            "retryable": False,
            "details": {"ledger_sequence": 12554, "transaction_hash": "abc123"},
        }
        rendered = json.dumps(public)
        assert "batch-9" not in rendered
        assert "row 5" not in rendered

    def test_public_dict_has_no_details_when_nothing_is_known(self) -> None:
        assert "details" not in ParseError("bad").to_public_dict()

    def test_ledger_zero_is_kept_in_the_public_details(self) -> None:
        assert ParseError("bad", ledger_sequence=0).to_public_dict()["details"] == {
            "ledger_sequence": 0
        }

    def test_field_name_alone_is_not_public(self) -> None:
        assert "details" not in ParseError("bad", field="created_at").to_public_dict()


class TestReferenceDocument:
    """``docs/v1-errors.md`` must list every public error so the reference cannot drift."""

    def test_the_reference_exists(self) -> None:
        assert ERRORS_DOC.is_file(), "docs/v1-errors.md is missing"

    @pytest.mark.parametrize("cls", PUBLIC_ERRORS)
    def test_every_public_error_is_documented_with_its_code(self, cls: type[AstroMLError]) -> None:
        text = ERRORS_DOC.read_text(encoding="utf-8")

        assert f"`{cls.__name__}`" in text
        assert f"`{cls.code}`" in text
