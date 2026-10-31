"""PII redaction service for LLM compliance logging (issue #412)."""

from __future__ import annotations

import re

# Patterns for detecting common PII
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "ssn": r"\b(?!000|666)[0-9]{3}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "api_key": r"(?i)(api[_-]?key|apikey|api[_-]?token|token)\s*[:=]\s*[a-zA-Z0-9_-]{20,}",
    "password": r"(?i)(password|passwd|pwd)\s*[:=]\s*[^\s]{8,}",
    "account_number": r"\b[0-9]{8,17}\b",
}


class PIIRedactor:
    """Service for detecting and redacting personally identifiable information."""

    def __init__(self, patterns: dict[str, str] = None) -> None:
        self.patterns = patterns or PII_PATTERNS

    def redact(self, text: str) -> tuple[str, dict[str, bool]]:
        """Redact PII from text and return redacted text plus detection results.

        All patterns are matched against the original, untouched text first,
        then replacements are applied in a single right-to-left pass. This
        avoids two correctness bugs that a naive "redact as you go" approach
        has: (1) span offsets going stale once an earlier replacement changes
        the string's length, which corrupts or leaks PII adjacent to an
        earlier redaction, and (2) two patterns matching overlapping spans
        (e.g. ``credit_card`` and ``account_number`` both matching the same
        digit run), which can mangle one redaction marker with another.

        Args:
            text: Text to redact

        Returns:
            Tuple of (redacted_text, pii_types_detected)
        """
        if not text:
            return text, {}

        pii_detected: dict[str, bool] = {}
        all_matches: list[tuple[int, int, str, str]] = []  # (start, end, pii_type, matched_text)

        for pii_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, text))
            pii_detected[pii_type] = bool(matches)
            for match in matches:
                start, end = match.span()
                all_matches.append((start, end, pii_type, match.group()))

        resolved_matches = self._resolve_overlaps(all_matches)

        redacted_text = text
        for start, end, pii_type, matched_text in sorted(
            resolved_matches, key=lambda m: m[0], reverse=True
        ):
            redacted = self._create_redaction(pii_type, matched_text)
            redacted_text = redacted_text[:start] + redacted + redacted_text[end:]

        return redacted_text, pii_detected

    @staticmethod
    def _resolve_overlaps(
        matches: list[tuple[int, int, str, str]],
    ) -> list[tuple[int, int, str, str]]:
        """Resolve overlapping matches from different PII patterns.

        When two patterns match overlapping spans, keep the longest match
        (ties broken by earliest start) and drop any match that overlaps a
        span already kept. This guarantees the final replacement pass never
        applies two redactions to the same character range.
        """
        # Longest span first, then earliest start, for stable, intuitive
        # precedence (e.g. a 16-digit credit card number wins over an
        # 8-17 digit "account number" pattern matching the same digits).
        ordered = sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0]))

        kept: list[tuple[int, int, str, str]] = []
        occupied: list[tuple[int, int]] = []
        for start, end, pii_type, matched_text in ordered:
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue
            kept.append((start, end, pii_type, matched_text))
            occupied.append((start, end))

        return kept

    @staticmethod
    def _create_redaction(pii_type: str, matched_text: str) -> str:
        """Create appropriate redaction for PII type.

        Args:
            pii_type: Type of PII detected
            matched_text: The matched text

        Returns:
            Redacted placeholder
        """
        if pii_type == "email":
            return "[EMAIL]"
        elif pii_type == "phone":
            return "[PHONE]"
        elif pii_type == "ssn":
            return "[SSN]"
        elif pii_type == "credit_card":
            return f"[CREDIT_CARD:****{matched_text[-4:]}]"
        elif pii_type == "ip_address":
            return "[IP_ADDRESS]"
        elif pii_type == "api_key":
            return "[API_KEY]"
        elif pii_type == "password":
            return "[PASSWORD]"
        elif pii_type == "account_number":
            return "[ACCOUNT_NUMBER]"
        else:
            return "[REDACTED]"

    def has_pii(self, text: str) -> bool:
        """Check if text contains any PII.

        Args:
            text: Text to check

        Returns:
            True if any PII detected
        """
        if not text:
            return False

        for pattern in self.patterns.values():
            if re.search(pattern, text):
                return True
        return False

    def get_pii_summary(self, text: str) -> dict[str, bool]:
        """Get summary of PII types detected in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with PII type as key and detection as boolean
        """
        _, pii_types = self.redact(text)
        return {k: v for k, v in pii_types.items() if v}


# Global PII redactor instance
pii_redactor = PIIRedactor()
