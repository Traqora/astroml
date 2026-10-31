"""Tests for LLM compliance/audit logging (issue #412, audited under #870).

The core security property under test: ``ComplianceLogger.log_interaction``
must never persist raw, unredacted PII into the audit log. This module
verifies that invariant holds for the single write path
``ComplianceLogger.log_interaction`` exposes — there is no other code path
in ``compliance_logger.py`` that constructs an ``LLMComplianceLog`` row, and
in particular no exception-handler or error path that logs the raw
prompt/response bypassing ``pii_redactor.redact(...)``.

These tests use a minimal fake ``AsyncSession`` rather than a real database,
since the property being verified is about *what gets redacted before the
ORM object is built*, not about persistence mechanics (which are already
covered by the ORM/integration layer elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from astroml.llm.compliance_logger import ComplianceLogger


@dataclass
class _FakeLogRecord:
    """Stand-in for ``api.models.orm.LLMComplianceLog`` that just captures
    the kwargs it was constructed with, so tests can assert on exactly what
    ``log_interaction`` was about to persist."""

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeAsyncSession:
    """Minimal async session double: records added objects, no-ops commit."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        return None

    async def refresh(self, obj: Any) -> None:
        return None


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> _FakeAsyncSession:
    """Swap the real ``LLMComplianceLog`` ORM class for the fake record so
    this test never needs a real database or the (unrelated, currently
    broken) ``astroml.db`` import chain."""
    import astroml.llm.compliance_logger as compliance_logger_module

    monkeypatch.setattr(compliance_logger_module, "LLMComplianceLog", _FakeLogRecord)
    return _FakeAsyncSession()


PII_MARKERS = [
    "mallory.test.user@example.com",
    "555-867-5309",
    "123-45-6789",
    "4111-1111-1111-1111",
]

PROMPT_WITH_PII = (
    f"My email is {PII_MARKERS[0]}, call me at {PII_MARKERS[1]}, "
    f"SSN {PII_MARKERS[2]}, card {PII_MARKERS[3]}."
)
RESPONSE_WITH_PII = f"Sure, I've noted your email {PII_MARKERS[0]} and card {PII_MARKERS[3]}."


@pytest.mark.asyncio
async def test_log_interaction_never_persists_raw_pii(fake_session: _FakeAsyncSession) -> None:
    """The invariant this audit is protecting: none of the raw PII marker
    strings we feed in may appear anywhere in the persisted log record."""
    logger = ComplianceLogger()

    await logger.log_interaction(
        session=fake_session,
        user_id=1,
        username="mallory",
        interaction_type="query",
        feature="fraud_explainer",
        prompt=PROMPT_WITH_PII,
        response=RESPONSE_WITH_PII,
        model_used="mock-model",
    )

    assert len(fake_session.added) == 1
    record = fake_session.added[0]

    persisted_text = f"{record.kwargs['prompt_redacted']} {record.kwargs['response_redacted']}"
    for marker in PII_MARKERS:
        assert marker not in persisted_text, f"raw PII leaked into audit log: {marker!r}"


@pytest.mark.asyncio
async def test_log_interaction_flags_pii_detected_and_types(
    fake_session: _FakeAsyncSession,
) -> None:
    logger = ComplianceLogger()

    record_result = await logger.log_interaction(
        session=fake_session,
        prompt=PROMPT_WITH_PII,
        response=RESPONSE_WITH_PII,
    )

    record = fake_session.added[0]
    assert record.kwargs["pii_detected"] is True
    assert record_result is not None  # log_interaction returns the record

    pii_types = record.kwargs["pii_types"]
    assert pii_types is not None
    # Email and credit card appear in both prompt and response.
    assert "prompt" in pii_types["email"]
    assert "response" in pii_types["email"]
    assert "prompt" in pii_types["credit_card"]
    assert "response" in pii_types["credit_card"]
    # SSN and phone only appear in the prompt.
    assert pii_types["ssn"] == ["prompt"]
    assert pii_types["phone"] == ["prompt"]


@pytest.mark.asyncio
async def test_log_interaction_without_pii_sets_pii_detected_false(
    fake_session: _FakeAsyncSession,
) -> None:
    logger = ComplianceLogger()

    await logger.log_interaction(
        session=fake_session,
        prompt="What is the current fraud risk score for this account?",
        response="The account has a low fraud risk score of 0.12.",
    )

    record = fake_session.added[0]
    assert record.kwargs["pii_detected"] is False
    assert record.kwargs["pii_types"] is None
    assert "fraud risk score" in record.kwargs["prompt_redacted"]


@pytest.mark.asyncio
async def test_log_interaction_redacts_even_on_error_status(
    fake_session: _FakeAsyncSession,
) -> None:
    """Regression guard for a real gap found while auditing this module: a
    failed interaction (status="error") is a very natural place for a
    caller to interpolate raw context into ``error_message`` for debugging
    (e.g. "failed to process input: <raw prompt>"), which would bypass
    redaction if ``error_message`` were persisted verbatim. ``error_message``
    is now redacted through the same ``pii_redactor`` as prompt/response —
    this pins that down so a future change can't silently regress it.
    """
    logger = ComplianceLogger()

    await logger.log_interaction(
        session=fake_session,
        prompt=PROMPT_WITH_PII,
        response="",
        status="error",
        error_message=f"Upstream LLM call failed for prompt containing {PII_MARKERS[0]}",
    )

    record = fake_session.added[0]
    assert record.kwargs["status"] == "error"
    assert PII_MARKERS[0] not in record.kwargs["prompt_redacted"]
    assert PII_MARKERS[0] not in record.kwargs["error_message"]
    assert "[EMAIL]" in record.kwargs["error_message"]
    assert "error_message" in record.kwargs["pii_types"]["email"]


@pytest.mark.asyncio
async def test_log_interaction_with_no_error_message_leaves_it_none(
    fake_session: _FakeAsyncSession,
) -> None:
    """``error_message=None`` (the default, used by successful interactions)
    must not be redacted into an empty string or otherwise mutated."""
    logger = ComplianceLogger()

    await logger.log_interaction(
        session=fake_session,
        prompt="Explain this transaction.",
        response="It is a standard payment operation.",
    )

    record = fake_session.added[0]
    assert record.kwargs["error_message"] is None
