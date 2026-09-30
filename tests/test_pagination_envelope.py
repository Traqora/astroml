"""Tests for the standard pagination envelope — issue #948.

`astroml/api/routers/accounts.py` and `astroml/api/routers/fraud.py` each
hand-roll the same `(items, total, page, page_size)` response shape and the
same `offset = (page - 1) * page_size` math for every paginated endpoint.
`PageParams`/`Page`/`paginate_offset` in `astroml.db.session` give call
sites one canonical envelope and one canonical offset computation instead.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astroml.db.session import Page, PageParams, paginate_offset

# ---------------------------------------------------------------------------
# PageParams
# ---------------------------------------------------------------------------


def test_page_params_defaults() -> None:
    params = PageParams()
    assert params.page == 1
    assert params.page_size == 20
    assert params.offset == 0
    assert params.limit == 20


@pytest.mark.parametrize(
    ("page", "page_size", "expected_offset"),
    [
        (1, 20, 0),
        (2, 20, 20),
        (3, 20, 40),
        (1, 100, 0),
        (5, 1, 4),
    ],
)
def test_page_params_offset_matches_router_formula(
    page: int, page_size: int, expected_offset: int
) -> None:
    """Must match the `(page - 1) * page_size` formula duplicated across routers."""
    params = PageParams(page=page, page_size=page_size)
    assert params.offset == expected_offset
    assert params.limit == page_size


def test_page_params_rejects_page_below_one() -> None:
    with pytest.raises(ValidationError):
        PageParams(page=0)


def test_page_params_rejects_negative_page() -> None:
    with pytest.raises(ValidationError):
        PageParams(page=-1)


def test_page_params_rejects_page_size_below_one() -> None:
    with pytest.raises(ValidationError):
        PageParams(page_size=0)


def test_page_params_rejects_page_size_above_cap() -> None:
    """Matches the existing routers' `le=100` cap on page_size."""
    with pytest.raises(ValidationError):
        PageParams(page_size=101)


def test_page_params_accepts_page_size_at_cap() -> None:
    params = PageParams(page_size=100)
    assert params.page_size == 100


# ---------------------------------------------------------------------------
# paginate_offset / Page
# ---------------------------------------------------------------------------


def test_paginate_offset_builds_envelope() -> None:
    params = PageParams(page=1, page_size=2)
    page = paginate_offset(["a", "b"], total=5, params=params)

    assert isinstance(page, Page)
    assert page.items == ["a", "b"]
    assert page.total == 5
    assert page.page == 1
    assert page.page_size == 2


def test_paginate_offset_accepts_any_sequence_type() -> None:
    params = PageParams(page=1, page_size=3)
    page = paginate_offset(("a", "b", "c"), total=3, params=params)
    assert page.items == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("total", "page_size", "expected_total_pages"),
    [
        (0, 20, 0),
        (1, 20, 1),
        (20, 20, 1),
        (21, 20, 2),
        (40, 20, 2),
        (41, 20, 3),
        (1, 1, 1),
        (100, 100, 1),
        (101, 100, 2),
    ],
)
def test_page_total_pages_ceil_division(
    total: int, page_size: int, expected_total_pages: int
) -> None:
    params = PageParams(page=1, page_size=page_size)
    page = paginate_offset([], total=total, params=params)
    assert page.total_pages == expected_total_pages


def test_page_has_next_true_when_more_pages_remain() -> None:
    params = PageParams(page=1, page_size=10)
    page = paginate_offset(list(range(10)), total=25, params=params)
    assert page.has_next is True
    assert page.has_previous is False


def test_page_has_next_false_on_last_page() -> None:
    params = PageParams(page=3, page_size=10)
    page = paginate_offset(list(range(5)), total=25, params=params)
    assert page.has_next is False
    assert page.has_previous is True


def test_page_has_previous_false_on_first_page() -> None:
    params = PageParams(page=1, page_size=10)
    page = paginate_offset(list(range(10)), total=10, params=params)
    assert page.has_previous is False


def test_page_empty_result_has_zero_total_pages_and_no_next_or_previous() -> None:
    params = PageParams(page=1, page_size=20)
    page = paginate_offset([], total=0, params=params)
    assert page.items == []
    assert page.total == 0
    assert page.total_pages == 0
    assert page.has_next is False
    assert page.has_previous is False


def test_page_single_page_has_no_next_or_previous() -> None:
    params = PageParams(page=1, page_size=20)
    page = paginate_offset(["only-row"], total=1, params=params)
    assert page.total_pages == 1
    assert page.has_next is False
    assert page.has_previous is False


def test_page_rejects_negative_total() -> None:
    with pytest.raises(ValidationError):
        Page[str](items=[], total=-1, page=1, page_size=20)


def test_paginate_offset_preserves_item_order() -> None:
    params = PageParams(page=1, page_size=5)
    ordered = ["e", "d", "c", "b", "a"]
    page = paginate_offset(ordered, total=5, params=params)
    assert page.items == ordered
