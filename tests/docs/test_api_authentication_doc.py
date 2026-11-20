"""Documentation contract for deployed inference authentication (issue #1246)."""

from __future__ import annotations

import re

from tests.docs._helpers import assert_sections, doc_text, repo_path

DOC = "docs/API_AUTHENTICATION.md"

EXPECTED_SECTIONS = [
    "Purpose",
    "Supported credentials",
    "Public paths",
    "Deployment boundary",
    "Failure responses",
    "Example inference request",
    "Operator checklist",
    "Related documentation",
    "Verification",
]

EXPECTED_CONTENT = [
    "POST /api/v1/fraud/score",
    "Authorization: Bearer",
    "POST /api/v1/auth/login",
    "POST /api/v1/auth/api-keys",
    "ak_",
    "AUTH_ENABLED=true",
    "Authentication required",
    "Invalid or expired token",
    "Missing required scopes",
    "Rate limit exceeded",
    "Too many failed authentication attempts. Please try again later.",
    "API_KEY_ROTATION.md",
    "tests/docs/test_api_authentication_doc.py",
]


def test_authentication_doc_exists_and_has_required_sections() -> None:
    assert repo_path(DOC).is_file(), f"{DOC} is missing"
    text = doc_text(DOC)
    assert_sections(text, EXPECTED_SECTIONS, DOC)


def test_authentication_doc_covers_the_documented_boundary() -> None:
    text = doc_text(DOC)
    for snippet in EXPECTED_CONTENT:
        assert snippet in text, f"{DOC} does not document '{snippet}'"


def test_every_configured_public_path_is_documented() -> None:
    config = doc_text("api/auth/config.py")
    block = config.split("PUBLIC_PATHS = frozenset(", 1)[1].split(")", 1)[0]
    configured = set(re.findall(r'"(/[^"]+)"', block))
    assert configured, "Could not read PUBLIC_PATHS from api/auth/config.py"

    text = doc_text(DOC)
    for path in configured:
        assert f"`{path}`" in text, f"Public path {path} is not documented in {DOC}"

    assert "POST /api/v1/fraud/score" not in configured
    assert "`POST /api/v1/fraud/score`" in text


def test_documents_production_auth_switch_and_jwt_secret_enforcement() -> None:
    text = doc_text(DOC)
    assert "AUTH_ENABLED=false" in text
    assert "fails fast" in text
    assert "JWT_SECRET_KEY" in text
    assert "does not add an external dependency" in text


def test_new_doc_is_linked_from_navigation() -> None:
    index = doc_text("docs/index.md")
    nav = doc_text("docs/mkdocs.yml")
    assert "API_AUTHENTICATION.md" in index
    assert "API_AUTHENTICATION.md" in nav
