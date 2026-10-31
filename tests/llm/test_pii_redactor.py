"""Tests for the PII redaction service (issue #412, audited under #870).

Covers every PII category ``PIIRedactor`` claims to detect, false-positive
avoidance on clearly-non-PII text, boundary and multi-instance edge cases,
and a regression test for a real redaction-corruption bug found while
auditing this module: when multiple PII types co-occur in the same string,
naively mutating the output text while computing match spans against the
original (pre-redaction) string lets later replacements land on stale
offsets, corrupting or fully exposing already-matched PII. See
``PIIRedactor._resolve_overlaps`` and the right-to-left single-pass
replacement in ``PIIRedactor.redact`` for the fix.
"""

from __future__ import annotations

from astroml.llm.pii_redactor import PIIRedactor, pii_redactor


def test_module_level_singleton_is_usable() -> None:
    """The shared ``pii_redactor`` instance should work out of the box."""
    redacted, detected = pii_redactor.redact("Email me at a@b.com")
    assert redacted == "Email me at [EMAIL]"
    assert detected["email"] is True


# ─── Per-category redaction ───────────────────────────────────────────────────


def test_redacts_email() -> None:
    redactor = PIIRedactor()
    text = "Reach out to jane.doe+alerts@astroml.io for support."
    redacted, detected = redactor.redact(text)
    assert "jane.doe+alerts@astroml.io" not in redacted
    assert "[EMAIL]" in redacted
    assert detected["email"] is True


def test_redacts_phone_number() -> None:
    redactor = PIIRedactor()
    text = "Call me at (555) 123-4567 tomorrow."
    redacted, detected = redactor.redact(text)
    assert "555) 123-4567" not in redacted
    assert "[PHONE]" in redacted
    assert detected["phone"] is True


def test_redacts_ssn() -> None:
    redactor = PIIRedactor()
    text = "SSN on file: 123-45-6789."
    redacted, detected = redactor.redact(text)
    assert "123-45-6789" not in redacted
    assert "[SSN]" in redacted
    assert detected["ssn"] is True


def test_redacts_credit_card_and_keeps_last_four() -> None:
    redactor = PIIRedactor()
    text = "Card on file: 4111-1111-1111-1111."
    redacted, detected = redactor.redact(text)
    assert "4111-1111-1111-1111" not in redacted
    assert "[CREDIT_CARD:****1111]" in redacted
    assert detected["credit_card"] is True


def test_redacts_ip_address() -> None:
    redactor = PIIRedactor()
    text = "Request originated from 192.168.1.42."
    redacted, detected = redactor.redact(text)
    assert "192.168.1.42" not in redacted
    assert "[IP_ADDRESS]" in redacted
    assert detected["ip_address"] is True


def test_redacts_api_key() -> None:
    # Deliberately NOT shaped like a real provider's key format (e.g.
    # Stripe's `sk_live_...`) - secret-scanning on this repo's remote
    # correctly flags that shape even in an obviously-synthetic fixture,
    # since the scanner can't tell a test string from a real leaked key by
    # shape alone. The redactor's own pattern is provider-agnostic (any
    # `api_key: <20+ chars>` assignment), so a generic placeholder exercises
    # the same code path without tripping push protection.
    redactor = PIIRedactor()
    text = "api_key: PLACEHOLDER0123456789abcdefghijklmnop"
    redacted, detected = redactor.redact(text)
    assert "PLACEHOLDER0123456789abcdefghijklmnop" not in redacted
    assert "[API_KEY]" in redacted
    assert detected["api_key"] is True


def test_redacts_password() -> None:
    redactor = PIIRedactor()
    text = "password: Sup3rSecretValue!"
    redacted, detected = redactor.redact(text)
    assert "Sup3rSecretValue!" not in redacted
    assert "[PASSWORD]" in redacted
    assert detected["password"] is True


def test_redacts_account_number() -> None:
    redactor = PIIRedactor()
    text = "Account 00458213761 was credited."
    redacted, detected = redactor.redact(text)
    assert "00458213761" not in redacted
    assert "[ACCOUNT_NUMBER]" in redacted
    assert detected["account_number"] is True


# ─── False-positive avoidance ─────────────────────────────────────────────────


def test_does_not_redact_clean_text() -> None:
    redactor = PIIRedactor()
    text = "The fraud detection model flagged three transactions as anomalous."
    redacted, detected = redactor.redact(text)
    assert redacted == text
    assert all(value is False for value in detected.values())


def test_does_not_redact_short_numbers() -> None:
    """Short numeric tokens below the account-number threshold are left alone."""
    redactor = PIIRedactor()
    text = "There were 42 transactions across 7 accounts in block 2024."
    redacted, detected = redactor.redact(text)
    assert redacted == text
    assert detected["account_number"] is False


def test_empty_and_none_like_input() -> None:
    redactor = PIIRedactor()
    assert redactor.redact("") == ("", {})
    assert redactor.has_pii("") is False
    assert redactor.get_pii_summary("") == {}


# ─── Boundary and multi-instance edge cases ───────────────────────────────────


def test_pii_at_string_boundaries() -> None:
    """PII at the very start and very end of the string is still matched."""
    redactor = PIIRedactor()
    assert redactor.redact("john@example.com")[0] == "[EMAIL]"
    assert redactor.redact("555-123-4567")[0] == "[PHONE]"


def test_multiple_instances_of_same_pii_type() -> None:
    redactor = PIIRedactor()
    text = "Contacts: alice@corp.com, bob@corp.com, carol@corp.com."
    redacted, detected = redactor.redact(text)
    assert "@corp.com" not in redacted
    assert redacted.count("[EMAIL]") == 3
    assert detected["email"] is True


def test_multiple_distinct_pii_types_in_one_string() -> None:
    """Regression test: multiple *different* PII types co-occurring must not
    corrupt each other's redaction (see module docstring)."""
    redactor = PIIRedactor()
    text = "Contact me at john@example.com or 555-123-4567."
    redacted, _ = redactor.redact(text)
    assert redacted == "Contact me at [EMAIL] or [PHONE]."
    # No raw fragments of either PII value may survive in the output.
    assert "555-123-4567" not in redacted
    assert "4567" not in redacted
    assert "john@example.com" not in redacted


def test_three_distinct_pii_types_do_not_corrupt_each_other() -> None:
    """Stronger regression case: SSN + email + credit card together.

    Before the overlap/stale-offset fix, redacting the credit card (whose
    span was computed against the *original* string) after the SSN and
    email replacements had already shortened the working string caused the
    credit card's trailing digits to leak into the output verbatim, right
    next to the `[CREDIT_CARD:...]` marker.
    """
    redactor = PIIRedactor()
    text = "SSN 123-45-6789 and email bob@test.com here, card 4111-1111-1111-1111 too"
    redacted, detected = redactor.redact(text)

    assert redacted == ("SSN [SSN] and email [EMAIL] here, card [CREDIT_CARD:****1111] too")
    assert "123-45-6789" not in redacted
    assert "bob@test.com" not in redacted
    assert "4111-1111-1111-1111" not in redacted
    # No stray raw digit run from the card should leak in either.
    assert "4111-1111-1" not in redacted
    assert detected["ssn"] is True
    assert detected["email"] is True
    assert detected["credit_card"] is True


def test_overlapping_patterns_prefer_the_more_specific_match() -> None:
    """A 16-digit credit card number also satisfies the generic 8-17 digit
    ``account_number`` pattern. The more specific / longer match (credit
    card) must win outright rather than the two redactions colliding and
    mangling each other's output markers."""
    redactor = PIIRedactor()
    text = "Card 4111111111111111 end"
    redacted, detected = redactor.redact(text)

    assert redacted == "Card [CREDIT_CARD:****1111] end"
    assert "[ACCOUNT_NUMBER]" not in redacted
    assert detected["credit_card"] is True


def test_redaction_is_idempotent_on_already_redacted_text() -> None:
    """Running redact twice should not change already-redacted output."""
    redactor = PIIRedactor()
    text = "Email alice@corp.com and card 4111-1111-1111-1111."
    once, _ = redactor.redact(text)
    twice, _ = redactor.redact(once)
    assert once == twice


# ─── has_pii / get_pii_summary helpers ────────────────────────────────────────


def test_has_pii_true_and_false() -> None:
    redactor = PIIRedactor()
    assert redactor.has_pii("no sensitive data in this sentence") is False
    assert redactor.has_pii("my ssn is 123-45-6789") is True


def test_get_pii_summary_only_includes_detected_types() -> None:
    redactor = PIIRedactor()
    text = "email alice@corp.com and ssn 123-45-6789"
    summary = redactor.get_pii_summary(text)
    assert summary == {"email": True, "ssn": True}
    assert "phone" not in summary
    assert "credit_card" not in summary
