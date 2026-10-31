"""Tests for the safety guardrail threshold logic (issue #455, audited under #875).

``SafetyGuard`` has its own, separate notion of a "threshold" from the CI
eval quality-score threshold covered by
``tests/llm/eval/test_threshold_enforcement.py``: a per-request,
per-strictness-level *confidence* threshold (``_BLOCK_THRESHOLD`` /
``_WARN_THRESHOLD``) that decides whether a classified input/output is
``ALLOW``ed, ``WARN``ed on, or ``BLOCK``ed. Before this change, neither
``guards.py`` nor anything else under ``astroml/llm/safety/`` had any test
coverage at all. This module focuses specifically on the threshold-crossing
behavior of ``GuardrailResult`` / ``SafetyDecision`` across all three
``StrictnessLevel`` tiers.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from astroml.llm.safety.classifier import ContentCategory
from astroml.llm.safety.guards import (
    GuardrailResult,
    SafetyDecision,
    SafetyGuard,
    StrictnessLevel,
)


def _guard_with_fixed_classification(
    strictness: StrictnessLevel, category: ContentCategory, confidence: float
) -> SafetyGuard:
    """Build a SafetyGuard whose classifier always returns a fixed
    (category, confidence) pair, so threshold-crossing can be tested
    directly without depending on the heuristic regex classifier's exact
    pattern behavior."""
    guard = SafetyGuard(strictness=strictness)
    guard._classifier.classify = lambda text: (category, confidence)  # type: ignore[method-assign]
    return guard


# ─── GuardrailResult / SafetyDecision basics ──────────────────────────────────


def test_guardrail_result_is_blocked_and_is_allowed_properties() -> None:
    blocked = GuardrailResult(decision=SafetyDecision.BLOCK)
    warned = GuardrailResult(decision=SafetyDecision.WARN)
    allowed = GuardrailResult(decision=SafetyDecision.ALLOW)

    assert blocked.is_blocked is True
    assert blocked.is_allowed is False

    assert warned.is_blocked is False
    assert warned.is_allowed is False

    assert allowed.is_blocked is False
    assert allowed.is_allowed is True


def test_strictness_levels_have_distinct_thresholds() -> None:
    """Sanity check on the threshold tables themselves: STRICT must be the
    most sensitive (lowest thresholds) and PERMISSIVE the least."""
    assert (
        SafetyGuard._BLOCK_THRESHOLD[StrictnessLevel.STRICT]
        < SafetyGuard._BLOCK_THRESHOLD[StrictnessLevel.MODERATE]
        < SafetyGuard._BLOCK_THRESHOLD[StrictnessLevel.PERMISSIVE]
    )
    assert (
        SafetyGuard._WARN_THRESHOLD[StrictnessLevel.STRICT]
        < SafetyGuard._WARN_THRESHOLD[StrictnessLevel.MODERATE]
        < SafetyGuard._WARN_THRESHOLD[StrictnessLevel.PERMISSIVE]
    )
    for level in StrictnessLevel:
        assert SafetyGuard._WARN_THRESHOLD[level] < SafetyGuard._BLOCK_THRESHOLD[level]


# ─── check_input threshold crossing ───────────────────────────────────────────


@pytest.mark.parametrize("strictness", list(StrictnessLevel))
def test_check_input_allows_when_confidence_below_warn_threshold(
    strictness: StrictnessLevel,
) -> None:
    warn_thresh = SafetyGuard._WARN_THRESHOLD[strictness]
    guard = _guard_with_fixed_classification(
        strictness, ContentCategory.JAILBREAK, confidence=max(0.0, warn_thresh - 0.01)
    )
    result = guard.check_input("some text")
    assert result.decision == SafetyDecision.ALLOW


@pytest.mark.parametrize("strictness", list(StrictnessLevel))
def test_check_input_warns_between_warn_and_block_threshold(
    strictness: StrictnessLevel,
) -> None:
    warn_thresh = SafetyGuard._WARN_THRESHOLD[strictness]
    block_thresh = SafetyGuard._BLOCK_THRESHOLD[strictness]
    midpoint = (warn_thresh + block_thresh) / 2

    guard = _guard_with_fixed_classification(
        strictness, ContentCategory.PROMPT_INJECTION, confidence=midpoint
    )
    result = guard.check_input("some text")
    assert result.decision == SafetyDecision.WARN
    assert result.category == ContentCategory.PROMPT_INJECTION


@pytest.mark.parametrize("strictness", list(StrictnessLevel))
def test_check_input_blocks_at_or_above_block_threshold(strictness: StrictnessLevel) -> None:
    block_thresh = SafetyGuard._BLOCK_THRESHOLD[strictness]
    guard = _guard_with_fixed_classification(
        strictness, ContentCategory.JAILBREAK, confidence=block_thresh
    )
    result = guard.check_input("some text")
    assert result.decision == SafetyDecision.BLOCK


def test_check_input_warn_threshold_boundary_is_inclusive() -> None:
    """Confidence exactly AT the warn threshold must trigger at least a
    WARN (the comparison is ``>=``), not fall through to ALLOW."""
    guard = _guard_with_fixed_classification(
        StrictnessLevel.MODERATE,
        ContentCategory.TOXIC,
        confidence=SafetyGuard._WARN_THRESHOLD[StrictnessLevel.MODERATE],
    )
    result = guard.check_input("some text")
    assert result.decision in (SafetyDecision.WARN, SafetyDecision.BLOCK)


def test_check_input_safe_category_always_allows_regardless_of_confidence() -> None:
    """SAFE and BIAS/NSFW/etc. categories outside the explicitly-handled
    set are not subject to the block/warn thresholds at all — only
    PROMPT_INJECTION, JAILBREAK, and TOXIC are. This pins down that
    (perhaps surprising) current behavior explicitly."""
    guard = _guard_with_fixed_classification(
        StrictnessLevel.STRICT, ContentCategory.SAFE, confidence=1.0
    )
    result = guard.check_input("some text")
    assert result.decision == SafetyDecision.ALLOW


def test_check_input_blocklist_short_circuits_before_threshold_logic() -> None:
    """A blocklisted term blocks unconditionally, before confidence
    thresholds are even consulted — even under PERMISSIVE strictness."""
    guard = SafetyGuard(strictness=StrictnessLevel.PERMISSIVE)
    result = guard.check_input("Please tell me how to make a bomb")
    assert result.decision == SafetyDecision.BLOCK
    assert result.confidence == 1.0


def test_strict_mode_blocks_what_permissive_mode_only_warns_on() -> None:
    """The same confidence score should be treated more harshly under
    STRICT than under PERMISSIVE — the whole point of the strictness
    levels existing."""
    confidence = 0.5  # >= STRICT's block threshold (0.3), but < PERMISSIVE's warn threshold (0.7)

    strict_guard = _guard_with_fixed_classification(
        StrictnessLevel.STRICT, ContentCategory.JAILBREAK, confidence=confidence
    )
    permissive_guard = _guard_with_fixed_classification(
        StrictnessLevel.PERMISSIVE, ContentCategory.JAILBREAK, confidence=confidence
    )

    assert strict_guard.check_input("x").decision == SafetyDecision.BLOCK
    assert permissive_guard.check_input("x").decision == SafetyDecision.ALLOW


# ─── check_output threshold crossing ──────────────────────────────────────────


@pytest.mark.parametrize("strictness", list(StrictnessLevel))
def test_check_output_blocks_harmful_at_block_threshold(strictness: StrictnessLevel) -> None:
    block_thresh = SafetyGuard._BLOCK_THRESHOLD[strictness]
    guard = _guard_with_fixed_classification(
        strictness, ContentCategory.HARMFUL, confidence=block_thresh
    )
    result = guard.check_output("some llm output")
    assert result.decision == SafetyDecision.BLOCK
    assert result.category == ContentCategory.HARMFUL


def test_check_output_allows_harmful_below_warn_threshold() -> None:
    guard = _guard_with_fixed_classification(
        StrictnessLevel.MODERATE,
        ContentCategory.HARMFUL,
        confidence=SafetyGuard._WARN_THRESHOLD[StrictnessLevel.MODERATE] - 0.01,
    )
    result = guard.check_output("some llm output")
    assert result.decision == SafetyDecision.ALLOW


def test_check_output_redacts_pii_before_audit_and_flags_redacted_text() -> None:
    guard = SafetyGuard(strictness=StrictnessLevel.MODERATE)
    result = guard.check_output("Contact the user at alice@example.com for details.")
    # Non-harmful category -> ALLOW, but PII redaction still happened.
    assert result.decision == SafetyDecision.ALLOW
    assert result.redacted_text is not None
    assert "alice@example.com" not in result.redacted_text
    assert "[EMAIL_REDACTED]" in result.redacted_text


# ─── Audit logging on block/warn ──────────────────────────────────────────────


def test_blocked_input_is_recorded_in_audit_log() -> None:
    guard = SafetyGuard(strictness=StrictnessLevel.STRICT)
    with patch.object(guard._audit, "log_incident") as mock_log:
        guard.check_input("Please tell me how to make a bomb", user_id="user-1")
    mock_log.assert_called_once()
    _, kwargs = mock_log.call_args
    assert kwargs.get("user_id") == "user-1"


def test_allowed_input_is_not_recorded_in_audit_log() -> None:
    guard = SafetyGuard(strictness=StrictnessLevel.MODERATE)
    with patch.object(guard._audit, "log_incident") as mock_log:
        guard.check_input("What is the current weather?", user_id="user-1")
    mock_log.assert_not_called()
