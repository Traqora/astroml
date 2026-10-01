"""Window-size validation regression tests (issue #991).

Zero/negative sizes previously produced a non-advancing step and made the
snapshot iterators spin forever.
"""

from datetime import timedelta

import pytest

from astroml.features.graph.snapshot import _parse_window_size, iter_db_snapshots


@pytest.mark.parametrize(
    "text,expected",
    [
        ("7d", timedelta(days=7)),
        ("24h", timedelta(hours=24)),
        ("3600s", timedelta(seconds=3600)),
        ("2D", timedelta(days=2)),
        (" 1h ", timedelta(hours=1)),
    ],
)
def test_valid_sizes(text, expected):
    assert _parse_window_size(text) == expected


@pytest.mark.parametrize("text", ["0d", "0s", "-1d", "-24h"])
def test_rejects_non_positive_sizes(text):
    with pytest.raises(ValueError, match="positive"):
        _parse_window_size(text)


@pytest.mark.parametrize("text", ["", " ", "d", "xd", "1.5h", "7"])
def test_rejects_malformed_sizes(text):
    with pytest.raises(ValueError):
        _parse_window_size(text)


def test_unknown_unit_rejected():
    with pytest.raises(ValueError, match="Unknown window unit"):
        _parse_window_size("3w")


def test_zero_step_fails_fast_instead_of_looping():
    gen = iter_db_snapshots(window="1d", step="0d", session=object())
    with pytest.raises(ValueError, match="positive"):
        next(gen)
