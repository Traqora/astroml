"""Autonomous multi-step agent execution loop.

:class:`AgentExecutor` implements the classic reason → act → observe cycle:

1. send the goal (plus the tool catalogue) to the LLM,
2. if the model requested tools, execute them and append the observations,
3. repeat until the model answers directly, the step budget is exhausted,
   or too many tool calls fail.

The loop is fully bounded by :class:`~astroml.agent.types.AgentConfig` and
always returns an :class:`~astroml.agent.types.AgentRunResult` carrying a
complete :class:`~astroml.agent.types.AgentTrace`.  Model and tool failures
are recorded in the trace rather than raised, which keeps autonomous runs
safe to schedule.

Both native function calling (``mode="native"``) and text based ReAct
reasoning (``mode="react"``) are supported; ``mode="auto"`` prefers native
tool calls and falls back to ReAct parsing when the model returns none.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ._async import run_sync
from .compression import CompressionStats, PromptCompressor, total_stats
from .llm import LLMProvider, LLMResponse
from .memory import ConversationMemory, Memory
from .planner import ReActParser, TaskPlanner
from .tools import ToolRegistry, result_to_message
from .types import (
    AgentConfig,
    AgentRunResult,
    AgentStep,
    AgentTrace,
    Message,
    StepStatus,
    ToolCall,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are AstroML's autonomous task agent, operating on a dynamic graph of "
    "Stellar blockchain transactions.\n"
    "Work in explicit steps: decide what you need, call a tool, inspect the "
    "observation, then continue. Prefer tools over guessing, never invent tool "
    "output, and stop as soon as you can answer.\n"
    "When you have the answer, reply with it directly and do not call further "
    "tools."
)

REACT_INSTRUCTIONS = (
    "To use a tool, reply exactly in this format:\n"
    "Thought: <short reasoning>\n"
    "Action: <tool name>\n"
    "Action Input: <JSON object of arguments>\n"
    "When you are finished, reply with:\n"
    "Thought: <short reasoning>\n"
    "Final Answer: <answer>"
)


def format_tool_catalogue(tools: Sequence[ToolSpec], *, compact: bool = False) -> str:
    """Render tool specs as a plain text catalogue for the system prompt.

    ``compact=True`` emits one ``name(argument: type, ...): description`` line
    per tool.  It carries exactly the same information as the default layout but
    drops the indentation and the ``arguments:`` continuation lines, which is a
    meaningful saving on prompts that expose a large tool set.
    """
    if not tools:
        return "No tools are available; answer from your own knowledge."
    blocks = ["Available tools:"]
    for spec in tools:
        properties = (dict(spec.parameters) or {}).get("properties") or {}
        rendered = ", ".join(
            f"{name}: {schema.get('type', 'any')}"
            for name, schema in properties.items()
        )
        if compact:
            blocks.append(f"- {spec.name}({rendered}): {spec.description}")
            continue
        blocks.append(f"- {spec.name}: {spec.description}")
        if rendered:
            blocks.append(f"  arguments: {rendered}")
    return "\n".join(blocks)


@dataclass
class _Turn:
    """Interpretation of a single model turn."""

    thought: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    source: str = "react"
    error: Optional[str] = None


class AgentExecutor:
    """Drives an LLM through bounded, tool-using autonomous runs.

    Args:
        llm: Provider used for every reasoning turn.
        tools: Registry of callable tools. An empty registry is allowed.
        config: Loop tunables (see :class:`~astroml.agent.types.AgentConfig`).
        memory: Conversation store. A fresh
            :class:`~astroml.agent.memory.ConversationMemory` is created when
            omitted; pass a persistent memory to carry context across runs.
        system_prompt: Override for :data:`DEFAULT_SYSTEM_PROMPT`.
        on_step: Optional callback invoked with every completed
            :class:`~astroml.agent.types.AgentStep`, for progress reporting.
        tool_names: Optional allow-list restricting the tools visible to the
            model (must exist in ``tools``).
        planner: Optional :class:`~astroml.agent.planner.TaskPlanner` used to
            pre-decompose the goal into steps before execution.
        compressor: Optional
            :class:`~astroml.agent.compression.PromptCompressor` applied to the
            prompt of every model call.  Memory and traces keep the full
            conversation; only what is sent to the provider is compressed.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
        *,
        memory: Optional[Memory] = None,
        system_prompt: Optional[str] = None,
        on_step: Optional[Callable[[AgentStep], None]] = None,
        tool_names: Optional[Sequence[str]] = None,
        planner: Optional[TaskPlanner] = None,
        compressor: Optional[PromptCompressor] = None,
    ) -> None:
        self.llm = llm
        registry = tools if tools is not None else ToolRegistry()
        self.tools = registry.filtered(tool_names) if tool_names else registry
        self.config = config or AgentConfig()
        self.memory = memory if memory is not None else ConversationMemory()
        self.system_prompt = (
            system_prompt or self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        self.on_step = on_step
        self.planner = planner
        self.compressor = compressor

    @property
    def tool_specs(self) -> List[ToolSpec]:
        """Specs of the tools currently visible to the model."""
        return self.tools.specs()

    # -- prompt construction -------------------------------------------
    def build_system_message(self) -> Message:
        """System message describing the agent's contract and its tools."""
        parts = [self.system_prompt]
        if self.config.include_tool_specs_in_prompt:
            parts.append(
                format_tool_catalogue(
                    self.tools.specs(),
                    compact=self.config.compact_tool_catalogue,
                )
            )
        if self.config.mode in ("auto", "react"):
            parts.append(REACT_INSTRUCTIONS)
        return Message.system("\n\n".join(parts))

    # -- response interpretation ---------------------------------------
    def interpret_response(self, response: LLMResponse) -> "_Turn":
        """Public wrapper around :meth:`_interpret` (convenient in tests)."""
        return self._interpret(response.content, response.tool_calls)

    def _interpret(self, content: str, native_calls: Sequence[ToolCall]) -> _Turn:
        """Turn a raw provider response into tool calls or a final answer."""
        text = (content or "").strip()
        prefer_native = self.config.mode == "native" or (
            self.config.mode == "auto" and bool(native_calls)
        )

        if prefer_native and native_calls:
            return _Turn(thought=text, calls=list(native_calls), source="native")

        if self.config.mode == "native":
            if text:
                return _Turn(thought=text, final_answer=text, source="native")
            return _Turn(error="model returned neither tool calls nor an answer")

        parsed = ReActParser.parse(text)
        if parsed.is_final:
            return _Turn(
                thought=parsed.thought,
                final_answer=parsed.final_answer,
                source="react",
            )
        if parsed.has_action:
            return _Turn(
                thought=parsed.thought,
                calls=[parsed.to_tool_call()],
                source="react",
            )
        if text:
            # Free-form prose with no markers is treated as the final answer.
            return _Turn(thought=parsed.thought, final_answer=text, source="react")
        return _Turn(error="model returned neither tool calls nor an answer")


    # -- execution ------------------------------------------------------
    async def arun(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
    ) -> AgentRunResult:
        """Run *goal* to completion, returning the answer and full trace.

        This method never raises for LLM or tool failures — inspect
        ``result.trace.status`` / ``result.trace.error`` for diagnostics.
        """
        if not goal or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        trace = AgentTrace(
            goal=goal,
            started_at=time.time(),
            status=StepStatus.RUNNING,
            metadata={
                "mode": self.config.mode,
                "provider": getattr(self.llm, "name", type(self.llm).__name__),
                "model": getattr(self.llm, "model", None),
                "tools": self.tools.names(),
            },
        )

        user_content = goal if not context else f"{goal}\n\nContext:\n{context}"
        user_content = await self._apply_planner(goal, user_content, context, trace)

        self.memory.extend([self.build_system_message(), Message.user(user_content)])

        tool_errors = 0
        compression: List[CompressionStats] = []

        for step_index in range(self.config.max_steps):
            step = AgentStep(index=step_index, status=StepStatus.RUNNING)

            prompt_messages = self.memory.messages()
            if self.compressor is not None:
                try:
                    compressed = self.compressor.compress(prompt_messages)
                except Exception as exc:  # noqa: BLE001 - compression is best effort
                    logger.warning(
                        "Prompt compression failed; sending the full prompt: %s", exc
                    )
                else:
                    prompt_messages = compressed.messages
                    compression.append(compressed.stats)
                    step.metadata["compression"] = compressed.stats.to_dict()

            try:
                response = await self.llm.complete(
                    prompt_messages,
                    tools=self.tools.specs(),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - reported via the trace
                logger.warning("LLM call failed: %s", exc)
                step.status = StepStatus.FAILED
                step.error = f"LLM call failed: {exc}"
                trace.steps.append(step)
                trace.status = StepStatus.FAILED
                trace.error = step.error
                break

            step.prompt_tokens = response.prompt_tokens
            step.completion_tokens = response.completion_tokens

            turn = self._interpret(response.content, response.tool_calls)
            step.thought = turn.thought
            step.metadata["source"] = turn.source

            if turn.error:
                step.status = StepStatus.FAILED
                step.error = turn.error
                trace.steps.append(step)
                trace.status = StepStatus.FAILED
                trace.error = turn.error
                break

            if turn.final_answer is not None:
                step.status = StepStatus.SUCCEEDED
                trace.steps.append(step)
                trace.final_answer = turn.final_answer
                trace.status = StepStatus.SUCCEEDED
                self.memory.add(Message.assistant(turn.final_answer))
                self._emit(step)
                break

            step.tool_calls = list(turn.calls)
            self.memory.add(
                Message.assistant(response.content, tool_calls=step.tool_calls)
            )

            # Use parallel execution when multiple tools are called and enabled
            if (
                self.config.parallel_tool_execution
                and len(step.tool_calls) >= self.config.parallel_tool_threshold
            ):
                results = await self.tools.run_calls_parallel(
                    step.tool_calls,
                    retries=self.config.tool_retries,
                    max_workers=self.config.max_parallel_workers,
                )
                for result in results:
                    step.results.append(result)
                    self.memory.add(result_to_message(result))
                    if not result.ok:
                        tool_errors += 1
            else:
                for call in step.tool_calls:
                    result: ToolResult = await self.tools.run_call(
                        call, retries=self.config.tool_retries
                    )
                    step.results.append(result)
                    self.memory.add(result_to_message(result))
                    if not result.ok:
                        tool_errors += 1

            step.status = StepStatus.FAILED if step.failed else StepStatus.SUCCEEDED
            trace.steps.append(step)
            self._emit(step)

            if self.config.stop_on_tool_error and step.failed:
                failed = next((item for item in step.results if not item.ok), None)
                trace.status = StepStatus.FAILED
                trace.error = (
                    f"Tool call failed: {failed.error}"
                    if failed is not None
                    else "Tool call failed"
                )
                break

            if tool_errors > self.config.max_tool_errors:
                trace.status = StepStatus.FAILED
                trace.error = (
                    f"Exceeded max_tool_errors={self.config.max_tool_errors} "
                    f"({tool_errors} failed tool calls)"
                )
                break
        else:
            trace.status = StepStatus.MAX_STEPS
            trace.error = (
                f"Stopped after max_steps={self.config.max_steps} "
                "without a final answer"
            )

        if compression:
            totals = total_stats(compression)
            trace.metadata["compression"] = {
                **totals.to_dict(),
                "turns": len(compression),
                "enabled": True,
            }

        trace.finished_at = time.time()
        return AgentRunResult(trace=trace, answer=trace.final_answer)


    def run(self, goal: str, *, context: Optional[str] = None) -> AgentRunResult:
        """Blocking wrapper around :meth:`arun`."""
        return run_sync(self.arun(goal, context=context))

    # -- internals ------------------------------------------------------
    async def _apply_planner(
        self,
        goal: str,
        user_content: str,
        context: Optional[str],
        trace: AgentTrace,
    ) -> str:
        """Pre-decompose *goal* with the optional planner and record the plan."""
        if self.planner is None:
            return user_content

        try:
            plan = await self.planner.aplan(
                goal, context=context, tool_names=self.tools.names()
            )
        except Exception as exc:  # noqa: BLE001 - planning is best effort
            logger.warning("Planner failed, continuing without a plan: %s", exc)
            return user_content

        trace.metadata["plan"] = plan.to_dict()
        if plan.is_empty:
            return user_content

        lines = [
            f"{index + 1}. {step.description}"
            + (f" (tool: {step.tool})" if step.tool else "")
            for index, step in enumerate(plan.steps)
        ]
        return f"{user_content}\n\nSuggested plan:\n" + "\n".join(lines)

    def _emit(self, step: AgentStep) -> None:
        """Invoke the ``on_step`` callback without letting it break the run."""
        if self.on_step is None:
            return
        try:
            self.on_step(step)
        except Exception:  # noqa: BLE001 - observer callbacks are best effort
            logger.exception("on_step callback raised; continuing run")
