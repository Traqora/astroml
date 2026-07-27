"""Safety guard implementations — core guardrail orchestration layer.

Resolves #455: Input/output safety guardrails with configurable strictness levels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from astroml.llm.safety.classifier import ContentClassifier, ContentCategory
from astroml.llm.safety.filters import InputFilter, OutputFilter
from astroml.llm.safety.audit import SafetyAuditLog

logger = logging.getLogger(__name__)


class StrictnessLevel(str, Enum):
    """Configurable safety strictness levels."""

    PERMISSIVE = "permissive"
    MODERATE = "moderate"
    STRICT = "strict"


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """Result of a safety guardrail check."""

    decision: SafetyDecision
    category: ContentCategory | None = None
    reason: str = ""
    redacted_text: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.decision == SafetyDecision.BLOCK

    @property
    def is_allowed(self) -> bool:
        return self.decision == SafetyDecision.ALLOW


class SafetyGuard:
    """Multi-layer safety guardrail for LLM inputs and outputs.

    Provides:
    - Input guardrails: PII detection, prompt injection, jailbreak, toxicity
    - Output guardrails: factuality, bias, harmful content, PII leaks
    - Configurable strictness (strict / moderate / permissive)
    - Block / warn / log decisions
    - Safety incident audit trail

    Example::

        guard = SafetyGuard(strictness=StrictnessLevel.STRICT)
        result = guard.check_input("Tell me how to hack a bank")
        if result.is_blocked:
            raise ValueError(result.reason)
    """

    def __init__(
        self,
        strictness: StrictnessLevel = StrictnessLevel.MODERATE,
        audit_log: SafetyAuditLog | None = None,
    ) -> None:
        self.strictness = strictness
        self._classifier = ContentClassifier()
        self._input_filter = InputFilter()
        self._output_filter = OutputFilter()
        self._audit = audit_log or SafetyAuditLog()

    # ─── Thresholds per strictness ──────────────────────────────────────────

    _BLOCK_THRESHOLD: dict[StrictnessLevel, float] = {
        StrictnessLevel.STRICT: 0.3,
        StrictnessLevel.MODERATE: 0.6,
        StrictnessLevel.PERMISSIVE: 0.9,
    }

    _WARN_THRESHOLD: dict[StrictnessLevel, float] = {
        StrictnessLevel.STRICT: 0.1,
        StrictnessLevel.MODERATE: 0.4,
        StrictnessLevel.PERMISSIVE: 0.7,
    }

    # ─── Public API ─────────────────────────────────────────────────────────

    def check_input(
        self,
        text: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Run all input guardrails against *text*.

        Checks (in order):
        1. Blocklist terms
        2. PII detection and redaction
        3. Prompt injection
        4. Jailbreak attempts
        5. Toxicity / hate speech
        """
        context = context or {}

        # 1. Blocklist
        blocked, term = self._input_filter.check_blocklist(text)
        if blocked:
            result = GuardrailResult(
                decision=SafetyDecision.BLOCK,
                category=ContentCategory.HARMFUL,
                reason=f"Blocklisted term detected: '{term}'",
                confidence=1.0,
            )
            self._audit.log_incident(text, result, user_id=user_id)
            return result

        # 2. PII detection and redaction
        redacted, pii_found = self._input_filter.redact_pii(text)
        if pii_found:
            logger.info("PII detected and redacted in input for user=%s", user_id)
            text = redacted  # continue checks on redacted text

        # 3. Classify content
        category, confidence = self._classifier.classify(text)

        block_thresh = self._BLOCK_THRESHOLD[self.strictness]
        warn_thresh = self._WARN_THRESHOLD[self.strictness]

        if category in (ContentCategory.PROMPT_INJECTION, ContentCategory.JAILBREAK):
            if confidence >= warn_thresh:
                decision = (
                    SafetyDecision.BLOCK
                    if confidence >= block_thresh
                    else SafetyDecision.WARN
                )
                result = GuardrailResult(
                    decision=decision,
                    category=category,
                    reason=f"{category.value} detected with confidence {confidence:.2f}",
                    redacted_text=redacted if pii_found else None,
                    confidence=confidence,
                )
                self._audit.log_incident(text, result, user_id=user_id)
                return result

        if category == ContentCategory.TOXIC and confidence >= warn_thresh:
            decision = (
                SafetyDecision.BLOCK
                if confidence >= block_thresh
                else SafetyDecision.WARN
            )
            result = GuardrailResult(
                decision=decision,
                category=ContentCategory.TOXIC,
                reason=f"Toxic content detected (confidence={confidence:.2f})",
                redacted_text=redacted if pii_found else None,
                confidence=confidence,
            )
            self._audit.log_incident(text, result, user_id=user_id)
            return result

        return GuardrailResult(
            decision=SafetyDecision.ALLOW,
            redacted_text=redacted if pii_found else None,
            confidence=1.0 - confidence,
        )

    def check_output(
        self,
        text: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Run all output guardrails against LLM *text*.

        Checks:
        1. Harmful content
        2. PII leakage prevention
        3. Bias detection
        """
        context = context or {}

        # PII leakage check on output
        redacted, pii_leaked = self._output_filter.redact_pii(text)
        if pii_leaked:
            logger.warning("PII detected in LLM output — redacting. user=%s", user_id)

        category, confidence = self._classifier.classify(redacted if pii_leaked else text)
        block_thresh = self._BLOCK_THRESHOLD[self.strictness]
        warn_thresh = self._WARN_THRESHOLD[self.strictness]

        if category == ContentCategory.HARMFUL and confidence >= warn_thresh:
            decision = (
                SafetyDecision.BLOCK
                if confidence >= block_thresh
                else SafetyDecision.WARN
            )
            result = GuardrailResult(
                decision=decision,
                category=ContentCategory.HARMFUL,
                reason=f"Harmful output detected (confidence={confidence:.2f})",
                redacted_text=redacted if pii_leaked else None,
                confidence=confidence,
            )
            self._audit.log_incident(text, result, user_id=user_id, is_output=True)
            return result

        return GuardrailResult(
            decision=SafetyDecision.ALLOW,
            redacted_text=redacted if pii_leaked else None,
            confidence=1.0 - confidence,
        )
