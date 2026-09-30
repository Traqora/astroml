"""Core data types for the AstroML LLM agent framework.

This module is dependency free (standard library only) so that the agent
loop, tool registry and trace objects can be imported and unit tested
without any particular LLM SDK or network access.

The vocabulary mirrors function-calling APIs:

* :class:`Message` — a single chat turn (system / user / assistant / tool)
* :class:`ToolCall` — a request from the model to invoke a tool
* :class:`ToolResult` — the observation produced by executing a tool call
* :class:`AgentStep` — one reason → act → observe cycle of the agent loop
* :class:`AgentTrace` — the full, serialisable record of a single run
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class Role(str, Enum):
    """Chat roles understood by :class:`~astroml.agent.llm.LLMProvider`."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepStatus(str, Enum):
    """Lifecycle status shared by steps, traces and run results."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    MAX_STEPS = "max_steps"


_EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}


def _loads_or_wrap(raw: str) -> Any:
    """Best effort JSON decoding of a model supplied argument string."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"input": raw}


_call_counter = itertools.count(1)


def _auto_id(name: str) -> str:
    """Generate a stable-looking synthetic id for tool calls without one."""
    return f"call_{name or 'tool'}_{next(_call_counter)}"


@dataclass(frozen=True)
class ToolSpec:
    """Provider neutral description of a tool exposed to an LLM.

    ``parameters`` is a JSON-Schema object describing the accepted arguments.
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable, provider neutral representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters) or dict(_EMPTY_SCHEMA),
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """Render as an OpenAI / Ollama style ``tools`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters) or dict(_EMPTY_SCHEMA),
            },
        }


@dataclass
class ToolCall:
    """A model request to execute a registered tool."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    def to_openai_dict(self) -> Dict[str, Any]:
        """Render as an OpenAI style tool call (arguments JSON encoded)."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, default=str),
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ToolCall":
        """Build a tool call from a flat *or* OpenAI style dictionary."""
        function = raw.get("function") if isinstance(raw, Mapping) else None
        if isinstance(function, Mapping):
            name = function.get("name", "")
            arguments: Any = function.get("arguments", {})
        else:
            name = raw.get("name", "")
            arguments = raw.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _loads_or_wrap(arguments)
        if not isinstance(arguments, Mapping):
            arguments = {"input": arguments}
        call_id = raw.get("id")
        return cls(
            id=str(call_id) if call_id else _auto_id(str(name)),
            name=str(name),
            arguments=dict(arguments),
        )


@dataclass
class Message:
    """A single chat message exchanged with an LLM."""

    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            self.role = Role(self.role)

    # -- constructors --------------------------------------------------
    @classmethod
    def system(cls, content: str, **metadata: Any) -> "Message":
        return cls(role=Role.SYSTEM, content=content, metadata=dict(metadata))

    @classmethod
    def user(cls, content: str, **metadata: Any) -> "Message":
        return cls(role=Role.USER, content=content, metadata=dict(metadata))

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        **metadata: Any,
    ) -> "Message":
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=list(tool_calls or []),
            metadata=dict(metadata),
        )

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Provider neutral, JSON serialisable representation."""
        payload: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return payload

    def to_openai_dict(self) -> Dict[str, Any]:
        """Render for OpenAI compatible chat completion APIs."""
        if self.role is Role.TOOL:
            payload: Dict[str, Any] = {"role": "tool", "content": self.content}
            if self.tool_call_id:
                payload["tool_call_id"] = self.tool_call_id
            if self.name:
                payload["name"] = self.name
            return payload

        payload = {"role": self.role.value, "content": self.content or ""}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai_dict() for call in self.tool_calls]
        return payload


@dataclass
class ToolResult:
    """Observation produced by executing a :class:`ToolCall`."""

    call_id: str
    name: str
    ok: bool = True
    content: str = ""
    error: Optional[str] = None
    data: Any = None
    duration_s: float = 0.0

    @classmethod
    def failure(
        cls,
        call_id: str,
        name: str,
        error: str,
        *,
        duration_s: float = 0.0,
    ) -> "ToolResult":
        """Build a failed result whose ``content`` is LLM readable."""
        return cls(
            call_id=call_id,
            name=name,
            ok=False,
            content=f"ERROR: {error}",
            error=str(error),
            duration_s=duration_s,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "call_id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "content": self.content,
            "duration_s": round(self.duration_s, 6),
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.data is not None:
            payload["data"] = self.data
        return payload


def _short(text: Optional[str], limit: int = 200) -> str:
    """Collapse whitespace and truncate *text* for human readable summaries."""
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
@dataclass
class AgentStep:
    """One reason → act → observe cycle of the agent loop."""

    index: int
    thought: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    results: List[ToolResult] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    @property
    def observation(self) -> str:
        """Concatenated tool output that is fed back to the model."""
        return "\n".join(result.content for result in self.results)

    @property
    def failed(self) -> bool:
        """True when at least one tool call in this step failed."""
        return any(not result.ok for result in self.results)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "index": self.index,
            "thought": self.thought,
            "status": self.status.value,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "results": [result.to_dict() for result in self.results],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if self.error:
            payload["error"] = self.error
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class AgentTrace:
    """Full, serialisable record of one agent run."""

    goal: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else self.started_at
        return end - self.started_at

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)

    @property
    def tool_error_count(self) -> int:
        return sum(
            1 for step in self.steps for result in step.results if not result.ok
        )

    @property
    def prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        """Human readable, single screen overview of the run."""
        lines = [f"goal: {_short(self.goal)}", f"status: {self.status.value}"]
        lines.append(
            f"steps: {len(self.steps)}  tool_calls: {self.tool_call_count}"
            f"  tool_errors: {self.tool_error_count}"
        )
        duration = self.duration_s
        if duration is not None:
            lines.append(f"duration: {duration:.3f}s")
        if self.error:
            lines.append(f"error: {_short(self.error)}")
        if self.final_answer:
            lines.append(f"answer: {_short(self.final_answer)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "goal": self.goal,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "tool_calls": self.tool_call_count,
            "tool_errors": self.tool_error_count,
            "total_tokens": self.total_tokens,
        }
        if self.final_answer is not None:
            payload["final_answer"] = self.final_answer
        if self.error is not None:
            payload["error"] = self.error
        duration = self.duration_s
        if duration is not None:
            payload["duration_s"] = round(duration, 6)
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class AgentRunResult:
    """Return value of :meth:`~astroml.agent.executor.AgentExecutor.run`."""

    trace: AgentTrace
    answer: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.trace.status is StepStatus.SUCCEEDED

    def to_dict(self) -> Dict[str, Any]:
        payload = self.trace.to_dict()
        payload["answer"] = self.answer
        payload["success"] = self.success
        return payload

    def __str__(self) -> str:
        return self.answer or self.trace.summary()


#: Accepted values for :attr:`AgentConfig.mode`.
VALID_MODES = ("auto", "native", "react")


@dataclass
class AgentConfig:
    """Tunables for the autonomous execution loop."""

    #: Hard cap on reason → act → observe cycles.
    max_steps: int = 8
    #: Give up once more than this many tool calls have failed.
    max_tool_errors: int = 3
    #: Sampling temperature forwarded to the provider.
    temperature: float = 0.0
    #: Optional completion token budget forwarded to the provider.
    max_tokens: Optional[int] = None
    #: ``native`` uses provider tool calling, ``react`` parses textual output,
    #: ``auto`` decides per response (native first, then ReAct fallback).
    mode: str = "auto"
    #: Override for the default system prompt.
    system_prompt: Optional[str] = None
    #: Extra attempts per tool call when the tool raises a retryable error.
    tool_retries: int = 0
    #: Abort the run as soon as a tool call fails.
    stop_on_tool_error: bool = False
    #: Append a rendered tool catalogue to the system prompt (needed by
    #: models that do not support native tool calling).
    include_tool_specs_in_prompt: bool = True
    #: Render that catalogue one line per tool instead of one block per tool.
    #: Same information, fewer prompt tokens.
    compact_tool_catalogue: bool = False
    #: Enable parallel execution of multiple tool calls in a single step.
    parallel_tool_execution: bool = True
    #: Minimum number of tool calls required to enable parallel execution.
    parallel_tool_threshold: int = 2
    #: Maximum number of worker threads for parallel tool execution.
    #: ``None`` uses the CPU count.
    max_parallel_workers: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tool_errors < 0:
            raise ValueError("max_tool_errors must be >= 0")
        if self.tool_retries < 0:
            raise ValueError("tool_retries must be >= 0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when provided")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.parallel_tool_threshold < 1:
            raise ValueError("parallel_tool_threshold must be >= 1")
        if self.max_parallel_workers is not None and self.max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be >= 1 when provided")



