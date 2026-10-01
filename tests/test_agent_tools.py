"""Tests for tool definition, schema generation and the tool registry."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from astroml.agent.tools import (
    Tool,
    ToolError,
    ToolRegistry,
    render_result,
    result_to_message,
    schema_from_signature,
    tool,
    tool_from_callable,
)
from astroml.agent.types import ToolCall


def _run(coro):
    """Execute a coroutine from a synchronous test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

class TestSchemaFromSignature:
    def test_maps_annotations_and_required_arguments(self):
        def fn(
            name: str,
            count: int = 1,
            ratio: Optional[float] = None,
            tags: Optional[List[str]] = None,
            weights: Optional[Dict[str, int]] = None,
            anything: Any = None,
        ) -> None:
            """Docstring."""

        schema = schema_from_signature(fn)
        assert schema["type"] == "object"
        assert schema["required"] == ["name"]
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["count"] == {"type": "integer"}
        assert schema["properties"]["ratio"] == {"type": "number"}
        assert schema["properties"]["tags"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert schema["properties"]["weights"] == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }
        assert schema["properties"]["anything"] == {}

    def test_omits_required_when_every_argument_has_a_default(self):
        schema = schema_from_signature(lambda value=None: None)
        assert "required" not in schema
        assert schema["properties"]["value"] == {}

    def test_skips_varargs_and_kwargs(self):
        def fn(a: int, *args: Any, **kwargs: Any) -> None:
            """Docstring."""

        schema = schema_from_signature(fn)
        assert list(schema["properties"]) == ["a"]

    def test_handles_unresolvable_forward_references(self):
        def fn(payload: "not_a_real_type") -> None:  # noqa: F821
            """Docstring."""

        schema = schema_from_signature(fn)
        assert schema["properties"]["payload"] == {}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestTool:
    def test_decorator_uses_docstring_as_description(self):
        @tool()
        def count_edges(edges: List[Dict[str, Any]]) -> int:
            """Count the transactions in a graph."""
            return len(edges)

        assert isinstance(count_edges, Tool)
        assert count_edges.name == "count_edges"
        assert count_edges.description == "Count the transactions in a graph."
        assert count_edges.parameters["required"] == ["edges"]

    def test_decorator_allows_overrides(self):
        @tool(name="other", description="custom")
        def original() -> None:
            """Original."""

        assert original.name == "other"
        assert original.description == "custom"

    def test_falls_back_to_generated_description(self):
        @tool()
        def undocumented() -> None:
            pass

        assert undocumented.description == "Call undocumented"

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            Tool(name="", description="d", func=lambda: None)

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError):
            Tool(name="t", description="d", func="not callable")

    def test_run_executes_sync_callable(self):
        tool_obj = tool_from_callable(lambda value: value * 2, name="double")
        assert _run(tool_obj.run(value=21)) == 42

    def test_run_awaits_async_callable(self):
        async def double(value: int) -> int:
            """Double the value."""
            return value * 2

        tool_obj = tool_from_callable(double)
        assert tool_obj.is_async is True
        assert _run(tool_obj.run(value=4)) == 8

    def test_spec_matches_parameters(self):
        tool_obj = tool_from_callable(lambda value: value, name="identity")
        spec = tool_obj.to_spec()
        assert spec.name == "identity"
        assert spec.parameters["required"] == ["value"]

    def test_tool_from_callable_passes_through_tool_instances(self):
        tool_obj = tool_from_callable(lambda: None, name="noop")
        assert tool_from_callable(tool_obj) is tool_obj

    def test_tool_from_callable_rejects_non_callables(self):
        with pytest.raises(TypeError):
            tool_from_callable("nope")


class TestRenderResult:
    def test_strings_pass_through(self):
        assert render_result("hello") == "hello"

    def test_mappings_are_json_encoded(self):
        assert render_result({"a": 1}) == '{"a": 1}'

    def test_unserialisable_values_fall_back_to_str(self):
        assert "object" in render_result(object())

    def test_truncates_long_output(self):
        rendered = render_result("x" * 50, limit=10)
        assert rendered.startswith("x" * 10)
        assert "truncated" in rendered


class TestResultToMessage:
    def test_wraps_result_as_tool_message(self):
        tool_obj = tool_from_callable(lambda: "ok", name="noop")
        result = _run(ToolRegistry([tool_obj]).run("noop", call_id="c1"))
        message = result_to_message(result)
        assert message.role.value == "tool"
        assert message.content == "ok"
        assert message.tool_call_id == "c1"
        assert message.name == "noop"


def _named(func, name: str) -> Tool:
    """Wrap *func* as a :class:`Tool` with an explicit name."""
    return tool_from_callable(func, name=name)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def _registry(self) -> ToolRegistry:
        def alpha(value: int) -> int:
            """Return the value unchanged."""
            return value

        def beta() -> str:
            """Return a constant."""
            return "beta"

        return ToolRegistry([alpha, beta])

    def test_registration_preserves_order(self):
        registry = self._registry()
        assert registry.names() == ["alpha", "beta"]
        assert len(registry) == 2
        assert "alpha" in registry
        assert "missing" not in registry

    def test_register_returns_tool(self):
        registry = ToolRegistry()
        assert isinstance(registry.register(_named(lambda: None, "noop")), Tool)

    def test_registering_duplicate_names_raises(self):
        registry = ToolRegistry()
        registry.register(_named(lambda: None, "noop"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_named(lambda: None, "noop"))

    def test_extend_returns_self(self):
        registry = ToolRegistry()
        assert registry.extend([_named(lambda: None, "noop")]) is registry
        assert len(registry) == 1

    def test_get_unknown_tool_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown tool"):
            self._registry().get("nope")

    def test_specs_follow_registration_order(self):
        assert [spec.name for spec in self._registry().specs()] == ["alpha", "beta"]

    def test_iteration_yields_tools(self):
        assert [item.name for item in self._registry()] == ["alpha", "beta"]

    def test_filtered_restricts_to_requested_tools(self):
        assert self._registry().filtered(["beta"]).names() == ["beta"]

    def test_filtered_rejects_unknown_names(self):
        with pytest.raises(KeyError, match="Unknown tool"):
            self._registry().filtered(["beta", "nope"])

    def test_unregister_removes_tool(self):
        registry = self._registry()
        registry.unregister("alpha")
        assert registry.names() == ["beta"]
        with pytest.raises(KeyError):
            registry.unregister("alpha")

    def test_repr_lists_names(self):
        assert "alpha" in repr(self._registry())


class TestToolRegistryRun:
    def test_unknown_tool_returns_failed_result(self):
        registry = ToolRegistry([_named(lambda: None, "noop")])
        result = _run(registry.run("ghost", {"a": 1}, call_id="c1"))
        assert result.ok is False
        assert "Unknown tool 'ghost'" in result.error
        assert result.call_id == "c1"

    def test_successful_run_captures_data_and_content(self):
        registry = ToolRegistry([_named(lambda value: {"value": value}, "value")])
        result = _run(registry.run("value", {"value": 3}))
        assert result.ok is True
        assert result.data == {"value": 3}
        assert result.content == '{"value": 3}'

    def test_missing_arguments_produce_invalid_argument_error(self):
        registry = ToolRegistry([_named(lambda value: value, "needs_value")])
        result = _run(registry.run("needs_value", {}))
        assert result.ok is False
        assert "Invalid arguments" in result.error

    def test_tool_exceptions_are_captured(self):
        def explode() -> None:
            """Always fails."""
            raise RuntimeError("kaboom")

        result = _run(ToolRegistry([explode]).run("explode"))
        assert result.ok is False
        assert "kaboom" in result.error

    def test_tool_error_is_reported_to_the_model(self):
        def guarded(value: int) -> int:
            """Reject negative values."""
            if value < 0:
                raise ToolError("value must be >= 0")
            return value

        result = _run(ToolRegistry([guarded]).run("guarded", {"value": -1}))
        assert result.ok is False
        assert result.content == "ERROR: value must be >= 0"

    def test_retryable_tool_error_is_retried(self):
        attempts = {"count": 0}

        def flaky() -> str:
            """Fail once, then succeed."""
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ToolError("temporary", retryable=True)
            return "recovered"

        result = _run(ToolRegistry([flaky]).run("flaky", retries=1))
        assert result.ok is True
        assert result.data == "recovered"
        assert attempts["count"] == 2

    def test_non_retryable_error_is_not_retried(self):
        attempts = {"count": 0}

        def failing() -> None:
            """Always fails without being retryable."""
            attempts["count"] += 1
            raise ToolError("permanent")

        result = _run(ToolRegistry([failing]).run("failing", retries=3))
        assert result.ok is False
        assert attempts["count"] == 1

    def test_run_call_uses_call_metadata(self):
        registry = ToolRegistry([_named(lambda: "ok", "value")])
        call = ToolCall(id="abc", name="value", arguments={})
        result = _run(registry.run_call(call))
        assert result.call_id == "abc"
        assert result.name == "value"
        assert result.ok is True
