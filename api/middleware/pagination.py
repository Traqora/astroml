"""Pagination limit helpers (issue #1247).

Provides ``clamp_page_size`` for use in route handlers and resolvers to
enforce the ``max_page_size`` setting and return a consistent error when
callers request more items than allowed.
"""

from __future__ import annotations

from fastapi import HTTPException

from api.config import Settings, settings as _default_settings


def clamp_page_size(
    requested: int,
    _settings: Settings | None = None,
) -> int:
    """Return ``requested`` if it is within the allowed range, or raise HTTP 400.

    Parameters
    ----------
    requested:
        The page size the caller requested.
    _settings:
        Settings instance to read ``max_page_size`` from.  Defaults to the
        application singleton; pass an explicit instance in tests.

    Raises
    ------
    HTTPException(400)
        When ``requested`` exceeds ``settings.max_page_size``.
    """
    cfg = _settings if _settings is not None else _default_settings
    if requested > cfg.max_page_size:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "PAGE_SIZE_TOO_LARGE",
                "message": (
                    f"Requested page size {requested} exceeds the maximum "
                    f"allowed page size of {cfg.max_page_size}."
                ),
                "details": [],
            },
        )
    return requested
