"""Reusable input validation decorators.

Provides decorator factories that enforce common input constraints
on function parameters, reducing duplicated validation logic across
services and API layers.

Example::

    @validate_not_none("start_ledger", "end_ledger")
    @validate_positive_int("batch_size")
    @validate_range("batch_size", start=1, end=10_000)
    def ingest(start_ledger: int, end_ledger: int, batch_size: int = 100) -> None:
        ...
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def validate_not_none(*param_names: str) -> Callable[[F], F]:
    """Ensure the given parameters are not ``None``.

    Raises:
        TypeError: If any of the named parameters is ``None``.
    """

    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in param_names:
                if bound.arguments.get(name) is None:
                    raise TypeError(f"{name} must not be None")
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_positive_int(*param_names: str) -> Callable[[F], F]:
    """Ensure the given parameters are positive integers.

    The check accepts ``int`` (or ``bool`` subclass) values that are
    strictly greater than zero.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is not positive (> 0).
    """

    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in param_names:
                value = bound.arguments.get(name)
                if value is not None and not isinstance(value, bool):
                    if not isinstance(value, int):
                        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
                    if value <= 0:
                        raise ValueError(f"{name} must be a positive integer, got {value}")
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_range(
    param_name: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> Callable[[F], F]:
    """Ensure a numeric parameter falls within an inclusive range.

    Either *start*, *end*, or both may be provided.  When omitted the
    corresponding bound is not checked.

    Raises:
        TypeError: If the value is not an ``int``.
        ValueError: If the value is outside the allowed range.
    """

    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param_name)
            if value is not None and not isinstance(value, bool):
                if not isinstance(value, int):
                    raise TypeError(f"{param_name} must be an integer, got {type(value).__name__}")
                if start is not None and value < start:
                    raise ValueError(f"{param_name} must be >= {start}, got {value}")
                if end is not None and value > end:
                    raise ValueError(f"{param_name} must be <= {end}, got {value}")
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
