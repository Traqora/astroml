"""Assertion generation for test cases.

Provides assertion templates and dynamic assertion generation
based on return types, exceptions, and expected behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AssertionTemplate(Enum):
    """Pre-defined assertion templates."""
    EQUAL = "assert {actual} == {expected}"
    NOT_EQUAL = "assert {actual} != {expected}"
    TRUE = "assert {actual} is True"
    FALSE = "assert {actual} is False"
    IS_NONE = "assert {actual} is None"
    IS_NOT_NONE = "assert {actual} is not None"
    IN = "assert {expected} in {actual}"
    NOT_IN = "assert {expected} not in {actual}"
    GREATER_THAN = "assert {actual} > {expected}"
    LESS_THAN = "assert {actual} < {expected}"
    GREATER_EQUAL = "assert {actual} >= {expected}"
    LESS_EQUAL = "assert {actual} <= {expected}"
    APPROX_EQUAL = "assert abs({actual} - {expected}) < {tolerance}"
    IS_INSTANCE = "assert isinstance({actual}, {expected})"
    LENGTH = "assert len({actual}) == {expected}"
    RAISES = "with pytest.raises({expected}):\n    {call}"
    NOT_RAISES = "with pytest.raises({expected}):\n    pytest.fail('Expected no exception')"


@dataclass
class AssertionGenerator:
    """Generates assertions based on return types and expected behavior."""

    return_type_hint: Optional[str] = None
    expected_exceptions: List[str] = field(default_factory=list)
    edge_case: bool = False

    def generate_assertions(
        self,
        actual: str,
        expected: Any = None,
        tolerance: float = 1e-6,
    ) -> List[str]:
        """Generate appropriate assertions for the given context."""
        assertions = []

        if self.return_type_hint:
            type_assertions = self._type_to_assertions(actual)
            assertions.extend(type_assertions)

        if expected is not None:
            assertions.append(
                AssertionTemplate.EQUAL.value.format(
                    actual=actual,
                    expected=repr(expected),
                )
            )

        if self.edge_case:
            edge_assertions = self._edge_case_assertions(actual)
            assertions.extend(edge_assertions)

        return assertions

    def generate_exception_assertion(
        self,
        call: str,
        exception_type: str = "ValueError",
    ) -> str:
        """Generate an assertion that expects an exception."""
        return AssertionTemplate.RAISES.value.format(
            expected=exception_type,
            call=call,
        )

    def generate_type_assertion(self, actual: str, expected_type: str) -> str:
        """Generate an isinstance assertion."""
        return AssertionTemplate.IS_INSTANCE.value.format(
            actual=actual,
            expected=expected_type,
        )

    def _type_to_assertions(self, actual: str) -> List[str]:
        type_map = {
            "int": [f"isinstance({actual}, int)"],
            "float": [f"isinstance({actual}, float)"],
            "str": [f"isinstance({actual}, str)"],
            "bool": [f"isinstance({actual}, bool)"],
            "list": [f"isinstance({actual}, list)"],
            "dict": [f"isinstance({actual}, dict)"],
            "tuple": [f"isinstance({actual}, tuple)"],
            "set": [f"isinstance({actual}, set)"],
            "pd.DataFrame": [f"isinstance({actual}, pd.DataFrame)"],
            "np.ndarray": [f"isinstance({actual}, np.ndarray)"],
        }
        for type_str, assertion in type_map.items():
            if self.return_type_hint and type_str in self.return_type_hint:
                return [f"assert {assertion[0]}"]
        return [f"assert {actual} is not None"]

    def _edge_case_assertions(self, actual: str) -> List[str]:
        return [
            f"assert {actual} is not None",
            f"assert {actual} is not False",
        ]
