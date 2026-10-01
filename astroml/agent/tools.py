"""Tool definition, registration and JSON-Schema generation.

Tools are the *act* half of the agent loop: a :class:`Tool` wraps a plain
Python callable together with the metadata an LLM needs in order to invoke
it.  :class:`ToolRegistry` owns the available tool set and guarantees that a
bad tool call never crashes the loop — failures are returned as a readable
:class:`~astroml.agent.types.ToolResult` instead.
"""
from __future__ import annotations

import inspect
import json
import logging
import time
import types as _pytypes
from collections import abc as _abc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .types import Message, ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

_PY_TO_JSON: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
}

_CONTAINER_ORIGINS = {
    list,
    tuple,
    set,
    frozenset,
    _abc.Sequence,
    _abc.Set,
    _abc.Iterable,
}

_MAPPING_ORIGINS = {dict, _abc.Mapping, _abc.MutableMapping}

_UNION_ORIGINS = {Union}
if hasattr(_pytypes, "UnionType"):  # PEP 604 ``X | None`` (Python 3.10+)
    _UNION_ORIGINS.add(_pytypes.UnionType)

_SIMPLE_NAMES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


class ToolError(Exception):
    """Raised by a tool to signal a recoverable, model-visible failure."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """Map a Python type annotation onto a JSON-Schema fragment."""
    if annotation is None or annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    if origin is not None:
        if origin in _UNION_ORIGINS:
            options = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(options) == 1:
                return _annotation_to_schema(options[0])
            return {"anyOf": [_annotation_to_schema(option) for option in options]}
        if origin in _CONTAINER_ORIGINS:
            args = get_args(annotation)
            schema: Dict[str, Any] = {"type": "array"}
            if args:
                schema["items"] = _annotation_to_schema(args[0])
            return schema
        if origin in _MAPPING_ORIGINS:
            args = get_args(annotation)
            if len(args) == 2:
                return {
                    "type": "object",
                    "additionalProperties": _annotation_to_schema(args[1]),
                }
            return {"type": "object"}
        annotation = origin

    if isinstance(annotation, type) and annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    if isinstance(annotation, str):
        name = annotation.strip().lower()
        if name in _SIMPLE_NAMES:
            return {"type": _SIMPLE_NAMES[name]}
        return {}
    return {}


def schema_from_signature(func: Callable[..., Any]) -> Dict[str, Any]:
    """Derive a JSON-Schema object from a callable's signature and hints.

    Unresolvable forward references degrade gracefully to permissive ``{}``
    property schemas instead of raising.
    """
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - forward references may not resolve
        hints = {}
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, parameter.annotation)
        properties[name] = _annotation_to_schema(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _first_line(text: Optional[str]) -> str:
    """Return the first non-empty line of *text*, stripped."""
    if not text:
        return ""
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass
class Tool:
    """A callable exposed to an LLM.

    Args:
        name: Unique identifier the model uses to invoke the tool.
        description: Natural language explanation shown to the model.
        func: The Python callable to execute.
        parameters: JSON-Schema for ``func``'s arguments; derived from the
            signature when omitted.
        requires_confirmation: Marks destructive calls that a host
            application may want to gate behind human approval.
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool name must be a non-empty string")
        if not callable(self.func):
            raise TypeError(f"Tool '{self.name}' func must be callable")
        if not self.parameters:
            self.parameters = schema_from_signature(self.func)

    @property
    def is_async(self) -> bool:
        """True when the wrapped callable is a coroutine function."""
        return inspect.iscoroutinefunction(self.func)

    def spec(self) -> ToolSpec:
        """Provider neutral description handed to the LLM."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=dict(self.parameters),
        )

    def to_spec(self) -> ToolSpec:
        """Alias of :meth:`spec` for readability at call sites."""
        return self.spec()

    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool, awaiting async callables transparently."""
        if self.is_async:
            return await self.func(**kwargs)
        value = self.func(**kwargs)
        if inspect.isawaitable(value):
            return await value
        return value


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    requires_confirmation: bool = False,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a function into a :class:`Tool`.

    Example::

        @tool(description="Count the transactions in a window")
        def count_edges(edges: List[Dict[str, Any]], start_ts: int) -> int:
            ...
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or getattr(func, "__name__", "tool"),
            description=description
            or _first_line(getattr(func, "__doc__", None))
            or f"Call {getattr(func, '__name__', 'tool')}",
            func=func,
            parameters=dict(parameters or {}),
            requires_confirmation=requires_confirmation,
        )

    return decorator


def tool_from_callable(
    func: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    requires_confirmation: bool = False,
) -> Tool:
    """Wrap an existing callable as a :class:`Tool`.

    Instances of :class:`Tool` are returned unchanged so callers can accept
    both decorated and decorated-free tools.
    """
    if isinstance(func, Tool):
        return func
    if not callable(func):
        raise TypeError("tool_from_callable expects a Tool or a callable")
    return Tool(
        name=name or getattr(func, "__name__", "tool"),
        description=description
        or _first_line(getattr(func, "__doc__", None))
        or f"Call {getattr(func, '__name__', 'tool')}",
        func=func,
        parameters=dict(parameters or {}),
        requires_confirmation=requires_confirmation,
    )


def render_result(value: Any, *, limit: int = 4000) -> str:
    """Render a tool return value as text for the model's context.

    Strings pass through untouched; anything else is JSON encoded with a
    ``default=str`` fallback so exotic objects never break the loop.
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if limit > 0 and len(text) > limit:
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"
    return text


def result_to_message(result: ToolResult) -> Message:
    """Wrap a tool result as a ``tool`` role message for the next turn."""
    return Message.tool(result.content, tool_call_id=result.call_id, name=result.name)


class ToolRegistry:
    """Ordered collection of :class:`Tool` objects.

    Registration order is preserved so prompts are deterministic, which keeps
    agent runs reproducible.
    """

    def __init__(self, tools: Optional[Iterable[Any]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for item in tools or []:
            self.register(item)

    # -- registration --------------------------------------------------
    def register(self, candidate: Any) -> Tool:
        """Register a :class:`Tool` or a raw callable and return the tool."""
        resolved = tool_from_callable(candidate)
        if resolved.name in self._tools:
            raise ValueError(f"Tool '{resolved.name}' is already registered")
        self._tools[resolved.name] = resolved
        return resolved

    def unregister(self, name: str) -> Tool:
        """Remove and return the tool called *name*."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}")
        return self._tools.pop(name)

    def extend(self, tools: Iterable[Any]) -> "ToolRegistry":
        """Register several tools; returns ``self`` for chaining."""
        for item in tools:
            self.register(item)
        return self

    # -- lookup --------------------------------------------------------
    def get(self, name: str) -> Tool:
        """Return the tool called *name* or raise :class:`KeyError`."""
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "none"
            raise KeyError(f"Unknown tool: {name!r}. Registered tools: {known}") from None

    def names(self) -> List[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def specs(self) -> List[ToolSpec]:
        """Specs for every registered tool, in registration order."""
        return [item.spec() for item in self._tools.values()]

    def filtered(self, names: Sequence[str]) -> "ToolRegistry":
        """Return a new registry restricted to *names* (order of *names*)."""
        wanted = list(dict.fromkeys(names))
        unknown = [name for name in wanted if name not in self._tools]
        if unknown:
            raise KeyError(f"Unknown tool(s): {', '.join(unknown)}")
        return ToolRegistry([self._tools[name] for name in wanted])

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry({self.names()!r})"

    # -- execution -----------------------------------------------------
    async def run(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        call_id: Optional[str] = None,
        retries: int = 0,
    ) -> ToolResult:
        """Execute tool *name* and always return a :class:`ToolResult`.

        Unknown tools, invalid arguments and tool exceptions are converted
        into failed results so the agent loop can feed the error back to the
        model instead of crashing.  ``retries`` only retries
        :class:`ToolError` instances flagged ``retryable=True``.
        """
        arguments = dict(arguments or {})
        identifier = call_id or f"call_{name}"
        resolved = self._tools.get(name)
        if resolved is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return ToolResult.failure(
                identifier,
                name,
                f"Unknown tool '{name}'. Available tools: {known}",
            )

        attempts = max(1, int(retries) + 1)
        started = time.perf_counter()
        error: Optional[BaseException] = None

        for attempt in range(attempts):
            try:
                value = await resolved.run(**arguments)
            except ToolError as exc:
                error = exc
                if exc.retryable and attempt + 1 < attempts:
                    logger.debug("Retrying tool %s after retryable error: %s", name, exc)
                    continue
                break
            except TypeError as exc:
                error = ToolError(f"Invalid arguments for tool '{name}': {exc}")
                break
            except Exception as exc:  # noqa: BLE001 - surfaced to the model
                logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
                error = exc
                break

            duration = time.perf_counter() - started
            return ToolResult(
                call_id=identifier,
                name=name,
                ok=True,
                content=render_result(value),
                data=value,
                duration_s=duration,
            )

        duration = time.perf_counter() - started
        return ToolResult.failure(identifier, name, str(error), duration_s=duration)

    async def run_call(
        self,
        call: ToolCall,
        *,
        retries: int = 0,
    ) -> ToolResult:
        """Convenience wrapper executing a :class:`ToolCall`."""
        return await self.run(
            call.name, call.arguments, call_id=call.id, retries=retries
        )

    async def run_calls_parallel(
        self,
        calls: Sequence[ToolCall],
        *,
        retries: int = 0,
        max_workers: Optional[int] = None,
    ) -> List[ToolResult]:
        """Execute multiple tool calls in parallel.

        Uses ThreadPoolExecutor to run independent tool calls concurrently.
        Beneficial when tools are IO-bound (e.g., network requests, file I/O).

        Args:
            calls: Sequence of tool calls to execute.
            retries: Retry attempts per tool call for retryable errors.
            max_workers: Maximum number of worker threads. ``None`` uses CPU count.

        Returns:
            List of results in the same order as input calls.
        """
        if not calls:
            return []
        if len(calls) < 3:  # Threshold below which overhead outweighs benefits
            return [
                await self.run_call(call, retries=retries) for call in calls
            ]

        async def run_single(call: ToolCall) -> ToolResult:
            return await self.run_call(call, retries=retries)

        # Since we're already in an async context, we can use asyncio.gather
        # for better performance than ThreadPoolExecutor for async tools
        import asyncio

        results = await asyncio.gather(
            *[run_single(call) for call in calls], return_exceptions=False
        )
        return list(results)

