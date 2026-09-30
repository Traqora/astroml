"""Tests for astroml.utils.validators — reusable input validation decorators."""

from __future__ import annotations

import pytest

from astroml.utils.validators import validate_not_none, validate_positive_int, validate_range


class TestValidateNotNull:
    """Tests for the @validate_not_none decorator."""

    def test_passes_when_all_params_present(self):
        @validate_not_none("x", "y")
        def add(x: int, y: int) -> int:
            return x + y

        assert add(1, 2) == 3

    def test_passes_with_keyword_args(self):
        @validate_not_none("x", "y")
        def add(x: int, y: int) -> int:
            return x + y

        assert add(x=1, y=2) == 3

    def test_raises_on_none_positional(self):
        @validate_not_none("x")
        def greet(x: str) -> str:
            return f"hello {x}"

        with pytest.raises(TypeError, match="x must not be None"):
            greet(None)

    def test_raises_on_none_keyword(self):
        @validate_not_none("x")
        def greet(x: str) -> str:
            return f"hello {x}"

        with pytest.raises(TypeError, match="x must not be None"):
            greet(x=None)

    def test_allows_zero_and_false(self):
        @validate_not_none("x")
        def identity(x):
            return x

        assert identity(0) == 0
        assert identity(False) is False

    def test_preserves_function_name(self):
        @validate_not_none("x")
        def my_function(x: int) -> int:
            return x

        assert my_function.__name__ == "my_function"


class TestValidatePositiveInt:
    """Tests for the @validate_positive_int decorator."""

    def test_accepts_positive_int(self):
        @validate_positive_int("n")
        def compute(n: int) -> int:
            return n * 2

        assert compute(5) == 10

    def test_accepts_default_value(self):
        @validate_positive_int("n")
        def compute(n: int = 10) -> int:
            return n

        assert compute() == 10

    def test_rejects_zero(self):
        @validate_positive_int("n")
        def compute(n: int) -> int:
            return n

        with pytest.raises(ValueError, match="n must be a positive integer, got 0"):
            compute(0)

    def test_rejects_negative(self):
        @validate_positive_int("n")
        def compute(n: int) -> int:
            return n

        with pytest.raises(ValueError, match="n must be a positive integer, got -5"):
            compute(-5)

    def test_rejects_float(self):
        @validate_positive_int("n")
        def compute(n: int) -> int:
            return n

        with pytest.raises(TypeError, match="n must be an integer, got float"):
            compute(1.5)

    def test_rejects_string(self):
        @validate_positive_int("n")
        def compute(n: int) -> int:
            return n

        with pytest.raises(TypeError, match="n must be an integer, got str"):
            compute("five")

    def test_skips_none_params(self):
        @validate_positive_int("n")
        def compute(n: int | None = None) -> int | None:
            return n

        assert compute(None) is None

    def test_multiple_params(self):
        @validate_positive_int("x", "y")
        def add(x: int, y: int) -> int:
            return x + y

        assert add(1, 2) == 3

        with pytest.raises(ValueError, match="y must be a positive integer"):
            add(1, 0)


class TestValidateRange:
    """Tests for the @validate_range decorator."""

    def test_within_range(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int) -> int:
            return n

        assert compute(50) == 50

    def test_at_lower_bound(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int) -> int:
            return n

        assert compute(1) == 1

    def test_at_upper_bound(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int) -> int:
            return n

        assert compute(100) == 100

    def test_below_lower_bound(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int) -> int:
            return n

        with pytest.raises(ValueError, match="n must be >= 1, got 0"):
            compute(0)

    def test_above_upper_bound(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int) -> int:
            return n

        with pytest.raises(ValueError, match="n must be <= 100, got 101"):
            compute(101)

    def test_open_start(self):
        @validate_range("n", end=100)
        def compute(n: int) -> int:
            return n

        assert compute(-1000) == -1000
        assert compute(100) == 100

    def test_open_end(self):
        @validate_range("n", start=1)
        def compute(n: int) -> int:
            return n

        assert compute(1) == 1
        assert compute(999999) == 999999

    def test_rejects_non_int(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int) -> int:
            return n

        with pytest.raises(TypeError, match="n must be an integer, got float"):
            compute(1.5)

    def test_skips_none(self):
        @validate_range("n", start=1, end=100)
        def compute(n: int | None = None) -> int | None:
            return n

        assert compute(None) is None

    def test_preserves_function_name(self):
        @validate_range("n", start=0, end=10)
        def my_func(n: int) -> int:
            return n

        assert my_func.__name__ == "my_func"


class TestDecoratorComposition:
    """Tests for composing multiple validators on one function."""

    def test_all_decorators_together(self):
        @validate_not_none("start", "end")
        @validate_positive_int("batch_size")
        @validate_range("batch_size", start=1, end=10_000)
        def process(start: int, end: int, batch_size: int = 100) -> tuple:
            return (start, end, batch_size)

        assert process(1, 100) == (1, 100, 100)
        assert process(1, 100, batch_size=50) == (1, 100, 50)

    def test_not_none_fails_first(self):
        @validate_not_none("start")
        @validate_positive_int("batch_size")
        def process(start: int, batch_size: int = 100) -> None:
            pass

        with pytest.raises(TypeError, match="start must not be None"):
            process(None, batch_size=1)

    def test_positive_int_fails(self):
        @validate_not_none("start")
        @validate_positive_int("batch_size")
        def process(start: int, batch_size: int = 100) -> None:
            pass

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            process(1, batch_size=0)

    def test_range_fails(self):
        @validate_not_none("start")
        @validate_positive_int("batch_size")
        @validate_range("batch_size", start=1, end=1000)
        def process(start: int, batch_size: int = 100) -> None:
            pass

        with pytest.raises(ValueError, match="batch_size must be <= 1000"):
            process(1, batch_size=5000)
