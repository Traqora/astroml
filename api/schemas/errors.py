"""Shared API error response schema.

Error responses conform to RFC 7807 ("Problem Details for HTTP APIs",
https://www.rfc-editor.org/rfc/rfc7807), served as ``application/problem+json``.
This gives clients a stable, standard shape regardless of which router raised
the failure, rather than a project-specific envelope.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

PROBLEM_CONTENT_TYPE = "application/problem+json"

#: Base URI for this API's problem types. Appending a `code` (e.g.
#: VALIDATION_ERROR) yields a stable, dereferenceable `type` value per
#: RFC 7807 section 3.1; the URI does not need to resolve to a document.
PROBLEM_TYPE_BASE = "https://astroml.dev/problems"


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details object.

    `type`/`title` identify the problem *category* (stable across
    occurrences); `status`/`detail`/`instance` describe *this* occurrence.
    Extension members (e.g. `code`, `details`) are additional properties
    per RFC 7807 section 3.2.
    """

    type: str = Field(
        default="about:blank",
        examples=["https://astroml.dev/problems/VALIDATION_ERROR"],
    )
    title: str = Field(..., examples=["Request validation failed"])
    status: int = Field(..., examples=[422])
    detail: str | None = Field(default=None, examples=["Request validation failed"])
    instance: str | None = Field(default=None, examples=["/api/v1/accounts/GABC.../fraud"])
    code: str = Field(..., examples=["VALIDATION_ERROR"])
    details: Any | None = Field(default=None)


def problem_type_uri(code: str) -> str:
    """Build a stable `type` URI for a given error `code`."""
    return f"{PROBLEM_TYPE_BASE}/{code}"


def problem_detail(
    *,
    code: str,
    title: str,
    status: int,
    detail: str | None = None,
    instance: str | None = None,
    details: Any | None = None,
) -> dict[str, Any]:
    """Build an RFC 7807 problem details payload.

    Omits `detail`/`instance`/`details` when not provided rather than
    emitting them as null, per RFC 7807's guidance that all members besides
    `type` are optional.
    """
    payload: dict[str, Any] = {
        "type": problem_type_uri(code),
        "title": title,
        "status": status,
        "code": code,
    }
    if detail is not None:
        payload["detail"] = detail
    if instance is not None:
        payload["instance"] = instance
    if details is not None:
        payload["details"] = details
    return payload
