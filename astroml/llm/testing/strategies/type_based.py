"""Type-based test generation strategy.

Generates property-based tests and boundary cases from
type signatures and type annotations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, get_type_hints

logger = logging.getLogger(__name__)


class TypeBasedStrategy:
    """Strategy that generates tests by analyzing type annotations."""

    BUILTIN_TYPES = {int, float, str, bool, bytes, list, dict, tuple, set}

    def __init__(self, type_hints: Dict[str, str]):
        self.type_hints = type_hints

    def suggest_hypothesis_strategies(self) -> Dict[str, str]:
        """Suggest hypothesis strategies for each parameter."""
        strategies = {}
        for name, hint in self.type_hints.items():
            strategy = self._type_to_strategy(hint)
            if strategy:
                strategies[name] = strategy
        return strategies

    def identify_boundary_values(self) -> Dict[str, List[Any]]:
        """Identify boundary values for each parameter type."""
        boundaries = {}
        for name, hint in self.type_hints.items():
            boundaries[name] = self._get_boundaries(hint)
        return boundaries

    def _type_to_strategy(self, hint: str) -> Optional[str]:
        mapping = {
            "int": "st.integers()",
            "float": "st.floats()",
            "str": "st.text()",
            "bool": "st.booleans()",
            "bytes": "st.binary()",
            "list": "st.lists(st.integers())",
            "dict": "st.dictionaries(st.text(), st.integers())",
            "tuple": "st.tuples(st.integers())",
            "set": "st.sets(st.integers())",
            "None": "st.none()",
            "Any": "st.integers() | st.text() | st.floats() | st.booleans()",
        }
        for key, strategy in mapping.items():
            if key in hint:
                return strategy
        return "st.integers()"

    def _get_boundaries(self, hint: str) -> List[Any]:
        boundaries_map = {
            "int": [0, 1, -1, 2**31 - 1, -(2**31), 2**63 - 1],
            "float": [0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
            "str": ["", "a", "A" * 1000, "  ", "\x00", "\n"],
            "bool": [True, False],
            "list": [[], [1], [1] * 1000, None],
            "dict": [{}, {"a": 1}, None],
            "bytes": [b"", b"\x00", b"\xff" * 100],
        }
        for key, values in boundaries_map.items():
            if key in hint:
                return values
        return [None]
