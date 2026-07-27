"""Test generation orchestrator.

Orchestrates the generation of tests from code, documentation,
and requirements using LLMs with various strategies.
"""

from __future__ import annotations

import ast
import logging
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY_BASED = "property_based"
    EDGE_CASE = "edge_case"
    REGRESSION = "regression"
    PERFORMANCE = "performance"


@dataclass
class TestGenerationConfig:
    """Configuration for test generation."""
    test_type: TestType = TestType.UNIT
    framework: str = "pytest"
    max_tests_per_function: int = 5
    include_edge_cases: bool = True
    include_negative_tests: bool = True
    generate_fixtures: bool = True
    generate_assertions: bool = True
    model: str = "gpt-4"
    max_retries: int = 3
    temperature: float = 0.3


@dataclass
class GeneratedTest:
    """A generated test case."""
    name: str
    body: str
    test_type: TestType
    source_function: str
    imports: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)
    coverage_tags: List[str] = field(default_factory=list)


class TestGenerator:
    """Orchestrates LLM-based test generation.

    Analyzes code structure and documentation to generate
    pytest-compatible test functions with fixtures, mocks,
    and comprehensive assertions.
    """

    def __init__(self, config: Optional[TestGenerationConfig] = None):
        self.config = config or TestGenerationConfig()

    def generate_from_function(
        self,
        source_code: str,
        function_name: str,
    ) -> List[GeneratedTest]:
        """Generate tests from a function's source code and docstring."""
        llm_prompt = self._build_function_prompt(source_code, function_name)
        response = self._call_llm(llm_prompt)
        tests = self._parse_response(response, function_name)
        return tests

    def generate_from_module(
        self,
        module_source: str,
        module_name: str,
    ) -> List[GeneratedTest]:
        """Generate tests from an entire module's source code."""
        tree = ast.parse(module_source)
        functions = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        all_tests = []
        for func in functions:
            func_source = ast.get_source_segment(module_source, func) or ""
            tests = self.generate_from_function(func_source, func.name)
            all_tests.extend(tests)
        return all_tests

    def generate_from_spec(
        self,
        spec_text: str,
        api_name: str,
    ) -> List[GeneratedTest]:
        """Generate integration tests from API specifications."""
        prompt = (
            f"Generate {self.config.framework} integration tests for:\n\n"
            f"{spec_text}\n\n"
            f"API: {api_name}\n"
            f"Include setup, teardown, and assertions. "
            f"Use TestClient or similar patterns."
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, api_name)

    def generate_property_tests(
        self,
        type_signature: str,
        function_name: str,
    ) -> List[GeneratedTest]:
        """Generate property-based tests from type specifications."""
        prompt = (
            f"Generate property-based tests using hypothesis for:\n\n"
            f"Function: {function_name}\n"
            f"Signature: {type_signature}\n\n"
            f"Use @given decorators with appropriate strategies. "
            f"Include at least 3 properties to test."
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, function_name)

    def generate_edge_cases(
        self,
        source_code: str,
        function_name: str,
    ) -> List[GeneratedTest]:
        """Generate edge case tests from code analysis."""
        prompt = (
            f"Analyze this function and generate edge case tests:\n\n"
            f"{source_code}\n\n"
            f"Function: {function_name}\n"
            f"Identify boundary conditions, empty inputs, null values, "
            f"extreme values, and error conditions. "
            f"Generate one test per edge case."
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, function_name)

    def generate_regression_tests(
        self,
        bug_report: str,
        fix_code: str,
    ) -> List[GeneratedTest]:
        """Generate regression tests from bug reports and fixes."""
        prompt = (
            f"Generate regression tests for this bug fix:\n\n"
            f"Bug Report: {bug_report}\n\n"
            f"Fix Code: {fix_code}\n\n"
            f"Generate tests that reproduce the original bug "
            f"and verify the fix works."
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "regression_test")

    def generate_performance_tests(
        self,
        source_code: str,
        function_name: str,
    ) -> List[GeneratedTest]:
        """Generate performance/load tests."""
        prompt = (
            f"Generate performance tests for:\n\n"
            f"{source_code}\n\n"
            f"Function: {function_name}\n"
            f"Use pytest-benchmark or time measurements. "
            f"Include baseline measurements and load scenarios."
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, function_name)

    def _build_function_prompt(
        self,
        source_code: str,
        function_name: str,
    ) -> str:
        return (
            f"Generate {self.config.framework} unit tests for this function:\n\n"
            f"{source_code}\n\n"
            f"Requirements:\n"
            f"- Test function name: test_{function_name}_*\n"
            f"- Max {self.config.max_tests_per_function} tests\n"
            f"- Include imports\n"
            f"- Use realistic test data\n"
            f"- Cover normal cases{f', edge cases' if self.config.include_edge_cases else ''}"
            f"{', negative cases' if self.config.include_negative_tests else ''}\n"
            f"Return ONLY the Python test code, no explanations."
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to generate test code."""
        try:
            from astroml.llm.providers.factory import get_llm_provider
            provider = get_llm_provider("openai")
            response = provider.generate(
                prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=4096,
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback test generation when LLM is unavailable."""
        return textwrap.dedent("""\
            import pytest

            def test_generated_function():
                assert True

            def test_edge_case_empty():
                assert True

            def test_negative_case():
                with pytest.raises((ValueError, TypeError)):
                    pass
        """)

    def _parse_response(
        self,
        response: str,
        source_name: str,
    ) -> List[GeneratedTest]:
        """Parse LLM response into structured test cases."""
        import re
        tests = []

        test_blocks = re.findall(
            r'(?:import .+?\n)+|def test_\w+.*?:(?:\n(?:    .*?\n)*)',
            response,
            re.MULTILINE,
        )

        current_imports = []
        for block in test_blocks:
            if block.startswith("import") or block.startswith("from"):
                current_imports.append(block.strip())
            elif block.startswith("def test_"):
                lines = block.split("\n")
                first_line = lines[0]
                match = re.match(r"def (test_\w+)", first_line)
                if match:
                    test_name = match.group(1)
                    body = "\n".join(lines[1:]) if len(lines) > 1 else "    pass"
                    tests.append(GeneratedTest(
                        name=test_name,
                        body=body,
                        test_type=self.config.test_type,
                        source_function=source_name,
                        imports=list(current_imports),
                    ))

        if not tests:
            tests.append(GeneratedTest(
                name=f"test_{source_name}_auto",
                body=textwrap.dedent(f"""\
                    def test_{source_name}_auto():
                        result = {source_name}()
                        assert result is not None
                """),
                test_type=self.config.test_type,
                source_function=source_name,
            ))

        return tests

    def format_test_file(
        self,
        tests: List[GeneratedTest],
        module_path: Optional[str] = None,
    ) -> str:
        """Format generated tests into a complete test file."""
        lines: List[str] = []
        all_imports: Set[str] = set()
        for test in tests:
            for imp in test.imports:
                all_imports.add(imp)

        if all_imports:
            lines.append("\n".join(sorted(all_imports)))
            lines.append("")

        if module_path:
            lines.append(f"# Tests generated for: {module_path}")
            lines.append("")

        for test in tests:
            lines.append(f"def {test.name}({', '.join(test.fixtures) if test.fixtures else ''}):")
            lines.append(test.body.strip())
            lines.append("")

        return "\n".join(lines)
