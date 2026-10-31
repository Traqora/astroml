"""Hyperparameter optimization with search space validation.

This module provides:
- SearchSpace: Validated hyperparameter search space definition
- HPOptimizer: Base optimizer interface
- Validation for categorical (list) and range (tuple) parameter types

Dependencies:
- pydantic: Configuration validation
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class SearchSpace(BaseModel):
    """Validated hyperparameter search space.

    A search space maps parameter names to either:
    - A list of categorical choices (e.g., ["adam", "sgd"])
    - A tuple of (min, max) for numeric ranges (e.g., (0.001, 0.1))

    Example:
        >>> space = SearchSpace(params={
        ...     "learning_rate": (0.001, 0.1),
        ...     "optimizer": ["adam", "sgd"],
        ...     "hidden_dims": [[64, 32], [128, 64]],
        ... })
    """

    params: dict[str, Any] = Field(description="Parameter name to search space mapping")

    @field_validator("params")
    @classmethod
    def validate_search_space(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that all parameter definitions are well-formed.

        Rules:
        - Keys must be non-empty strings
        - Values must be either non-empty lists or 2-element tuples
        - Range tuples must have numeric min < max
        """
        if not v:
            raise ValueError("search space must not be empty")

        for key, value in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"parameter name must be a non-empty string, got {key!r}")

            if isinstance(value, list):
                if not value:
                    raise ValueError(f"parameter '{key}' list must not be empty")
            elif isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(
                        f"parameter '{key}' tuple must have exactly 2 elements (min, max), got {len(value)}"
                    )
                min_val, max_val = value
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    raise ValueError(
                        f"parameter '{key}' range values must be numeric, got {type(min_val).__name__} and {type(max_val).__name__}"
                    )
                if min_val >= max_val:
                    raise ValueError(
                        f"parameter '{key}' min must be less than max, got {min_val} >= {max_val}"
                    )
            else:
                raise ValueError(
                    f"parameter '{key}' must be a list (categorical) or tuple (range), got {type(value).__name__}"
                )

        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchSpace:
        """Create search space from dictionary with validation.

        Args:
            data: Raw search space dictionary.

        Returns:
            Validated SearchSpace instance.

        Raises:
            ValueError: If the search space is invalid.
        """
        try:
            return cls(params=data)
        except ValidationError as e:
            raise ValueError(f"Invalid search space: {e}") from e


class HPOptimizer:
    """Base hyperparameter optimizer interface."""

    def __init__(self, search_space: dict[str, Any] | SearchSpace | None = None) -> None:
        """Initialize optimizer with optional search space.

        Args:
            search_space: Either a raw dict or validated SearchSpace.
        """
        if search_space is not None:
            if isinstance(search_space, dict):
                self.search_space = SearchSpace.from_dict(search_space)
            else:
                self.search_space = search_space
        else:
            self.search_space = None

    def optimize(self) -> None:
        """Run hyperparameter optimization.

        Subclasses should implement this method.
        """
        pass
