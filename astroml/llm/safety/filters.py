"""Input and output filters — PII redaction and content filtering.

Resolves #455: Regex-based PII detection, redaction, and blocklist checking.
Supports: email, phone, SSN, credit card, IP address, physical addresses.
"""
from __future__ import annotations

import re
from typing import Sequence

from astroml.llm.safety.blocklist import BlocklistManager

# ─── PII Patterns ───────────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("ZIP_CODE", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("DATE_OF_BIRTH", re.compile(r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-]\d{4}\b")),
]


class InputFilter:
    """Filter and redact sensitive content from LLM inputs.

    Example::

        f = InputFilter()
        clean, pii_found = f.redact_pii("Email me at alice@example.com")
        # clean == "Email me at [EMAIL_REDACTED]", pii_found == True
    """

    def __init__(self, blocklist: BlocklistManager | None = None) -> None:
        self._blocklist = blocklist or BlocklistManager()

    def redact_pii(self, text: str) -> tuple[str, bool]:
        """Redact PII from *text*.

        Returns:
            (redacted_text, pii_was_found)
        """
        found = False
        for label, pattern in _PII_PATTERNS:
            replaced = pattern.sub(f"[{label}_REDACTED]", text)
            if replaced != text:
                text = replaced
                found = True
        return text, found

    def check_blocklist(self, text: str) -> tuple[bool, str]:
        """Check if *text* contains any blocklisted term.

        Returns:
            (is_blocked, matched_term_or_empty)
        """
        return self._blocklist.contains(text)

    def sanitize(self, text: str) -> str:
        """Strip dangerous HTML/script injections from text."""
        # Remove script tags and common injection vectors
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.I | re.S)
        text = re.sub(r"javascript:", "", text, flags=re.I)
        return text.strip()


class OutputFilter:
    """Filter and redact sensitive content from LLM outputs.

    Shares the same PII redaction logic as InputFilter but applied
    to the LLM's generated response to prevent data leakage.
    """

    def redact_pii(self, text: str) -> tuple[str, bool]:
        """Redact PII from LLM output *text*."""
        found = False
        for label, pattern in _PII_PATTERNS:
            replaced = pattern.sub(f"[{label}_REDACTED]", text)
            if replaced != text:
                text = replaced
                found = True
        return text, found

    def filter_harmful(self, text: str, patterns: Sequence[re.Pattern]) -> tuple[str, bool]:
        """Apply harmful-content filters to *text*.

        Returns:
            (filtered_text, content_was_removed)
        """
        filtered = False
        for p in patterns:
            replaced = p.sub("[CONTENT_FILTERED]", text)
            if replaced != text:
                text = replaced
                filtered = True
        return text, filtered
