"""Documentation contract for common API errors (issue #1243)."""

from __future__ import annotations

from tests.docs._helpers import assert_sections, doc_text, repo_path

DOC = "docs/API_ERRORS.md"

EXPECTED_SECTIONS = [
    "Purpose",
    "Response shape",
    "Common codes",
    "Authentication and rate limit messages",
    "Operator actions by symptom",
    "Logging safety",
    "Related documentation",
    "Verification",
]

EXPECTED_CODES = [
    "HTTP_400",
    "HTTP_401",
    "HTTP_403",
    "HTTP_404",
    "HTTP_409",
    "HTTP_429",
    "HTTP_503",
    "VALIDATION_ERROR",
    "INTERNAL_SERVER_ERROR",
]

EXPECTED_MESSAGES = [
    "Authentication required",
    "Invalid or expired token",
    "Admin privileges required",
    "Missing required scopes",
    "Rate limit exceeded",
    "Too many failed authentication attempts. Please try again later.",
    "Scoring service temporarily unavailable",
    "Request validation failed",
    "Internal server error",
]


def test_error_doc_exists_and_has_required_sections() -> None:
    assert repo_path(DOC).is_file(), f"{DOC} is missing"
    assert_sections(doc_text(DOC), EXPECTED_SECTIONS, DOC)


def test_error_doc_covers_every_stable_code() -> None:
    text = doc_text(DOC)
    for code in EXPECTED_CODES:
        assert f"`{code}`" in text, f"Stable code {code} is not documented in {DOC}"


def test_error_doc_matches_the_error_source_of_truth() -> None:
    schema = doc_text("api/schemas/errors.py")
    assert "PROBLEM_TYPE_BASE" in schema
    assert "https://astroml.dev/problems" in schema

    middleware = doc_text("api/middleware/errors.py")
    assert "VALIDATION_ERROR" in middleware
    assert "INTERNAL_SERVER_ERROR" in middleware
    assert "unhandled_exception_handler" in middleware

    text = doc_text(DOC)
    assert "https://astroml.dev/problems/<code>" in text
    assert "application/problem+json" in text
    assert "`code`, not on `title` or `detail`" in text


def test_error_doc_covers_auth_messages_and_operator_actions() -> None:
    text = doc_text(DOC)
    for message in EXPECTED_MESSAGES:
        assert message in text, f"Message '{message}' is not documented in {DOC}"
    assert "Operator action" in text
    assert "Retry-After" in text
    assert "correlation id" in text


def test_new_doc_is_linked_from_navigation() -> None:
    index = doc_text("docs/index.md")
    nav = doc_text("docs/mkdocs.yml")
    assert "API_ERRORS.md" in index
    assert "API_ERRORS.md" in nav
