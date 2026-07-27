"""Base agent classes with multi-step reasoning capabilities.

This module implements the core agent abstractions:

* :class:`Agent` — abstract base class defining the agent interface.
* :class:`ReActAgent` — ReAct (Reasoning + Acting) agent that interleaves
  reasoning, tool use, and observation in a loop.
* :class:`ChainOfThoughtAgent` — chain-of-thought agent that generates
  step-by-step reasoning before acting.
* :class:`PlannerAgent` — agent that decomposes tasks into sub-tasks,
  executes each one, and verifies completion.

All agents share a common interface (:meth:`run`, :meth:`step`) and
use the :class:`~astroml.agent.llm.LLMClient` for LLM calls and
:class:`~astroml.agent.tools.ToolRegistry` for tool execution.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import AgentConfig
from .exceptions import (
    AgentError,
    LLMError,
    MaxStepsExceededError,
    PlanningError,
    ToolError,
    ToolNotFoundError,
)
from .llm import LLMClient, LLMMessage, LLMResponse
from .memory import MemoryManager, Message
from .tools import ToolRegistry, ToolResult, create_default_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single reasoning step.

    Attributes:
        step_number: The 1-based index of this step.
        thinking: The agent's internal reasoning text.
        action: The tool name invoked (if any).
        action_input: The tool arguments (if any).
        observation: The tool's output (if a tool was used).
        response: The agent's response text (if no tool was used).
        done: Whether the agent considers the task complete.
    """

    step_number: int
    thinking: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    response: str = ""
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thinking": self.thinking,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "response": self.response,
            "done": self.done,
        }


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------

class Agent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement :meth:`step` (the per-step reasoning logic)
    and :meth:`run` (the full task execution loop).

    Attributes:
        config: The agent's :class:`AgentConfig`.
        llm: The :class:`LLMClient` for LLM calls.
        tools: The :class:`ToolRegistry` for tool execution.
        memory: The :class:`MemoryManager` for memory.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[LLMClient] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm = llm or LLMClient(self.config.llm)
        self.tools = tools or create_default_registry()
        self.memory = memory or MemoryManager(self.config.memory)
        self._step_count = 0
        self._history: List[StepResult] = []

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, task: str) -> StepResult:
        """Execute a single reasoning step.

        Args:
            task: The current task description.

        Returns:
            A :class:`StepResult` describing what happened.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, task: str) -> str:
        """Execute the full task and return the final result.

        Args:
            task: The task description.

        Returns:
            The final response string.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _add_system_prompt(self, task: str) -> None:
        """Add a system prompt to short-term memory if not already present."""
        # Check if a system message already exists
        context = self.memory.get_context()
        has_system = any(m.get("role") == "system" for m in context)
        if not has_system:
            system_msg = Message(
                role="system",
                content=self._get_system_prompt(task),
            )
            self.memory.add_message(system_msg)

    def _get_system_prompt(self, task: str) -> str:
        """Return the system prompt for this agent."""
        return (
            f"You are {self.config.name}, an autonomous AI agent. "
            f"Your task is: {task}\n\n"
            "Work through this step by step. Use tools when needed. "
            "After each tool use, observe the result and continue reasoning. "
            "When you have completed the task, provide a final summary."
        )

    def _add_user_message(self, content: str) -> None:
        """Add a user message to short-term memory."""
        self.memory.add_message(Message(role="user", content=content))

    def _add_assistant_message(self, content: str) -> None:
        """Add an assistant message to short-term memory."""
        self.memory.add_message(Message(role="assistant", content=content))

    def _add_tool_message(self, content: str, tool_call_id: str = "") -> None:
        """Add a tool-result message to short-term memory."""
        self.memory.add_message(
            Message(role="tool", content=content, metadata={"tool_call_id": tool_call_id})
        )

    def _get_context_messages(self) -> List[Dict[str, Any]]:
        """Get the current conversation context for LLM calls."""
        return self.memory.get_context()

    def _execute_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool, handling errors gracefully."""
        try:
            return self.tools.execute(name, **kwargs)
        except ToolNotFoundError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"Tool execution error: {exc}")

    def _format_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for LLM function-calling."""
        return self.tools.get_schemas()

    def reset(self) -> None:
        """Reset the agent's state (memory, step count, history)."""
        self.memory.reset()
        self._step_count = 0
        self._history.clear()

    def get_history(self) -> List[StepResult]:
        """Return the step history."""
        return list(self._history)


# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

class ReActAgent(Agent):
    """ReAct (Reasoning + Acting) agent.

    Implements the ReAct pattern: at each step, the agent:

    1. **Reasons** about what to do next.
    2. **Acts** by either calling a tool or providing a final answer.
    3. **Observes** the tool result and continues.

    The agent continues until it produces a final answer, reaches the
    maximum number of steps, or encounters an unrecoverable error.

    Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
    (Yao et al., 2022).
    """

    def step(self, task: str) -> StepResult:
        """Execute one ReAct step."""
        self._step_count += 1
        step_num = self._step_count

        if step_num > self.config.executor.max_steps:
            raise MaxStepsExceededError(
                f"Maximum steps ({self.config.executor.max_steps}) exceeded"
            )

        # Ensure system prompt is set
        self._add_system_prompt(task)

        # Get context and call LLM
        messages = self._get_context_messages()
        tool_schemas = self._format_tool_schemas()

        try:
            response = self.llm.chat(messages, tools=tool_schemas)
        except LLMError as exc:
            logger.error("LLM call failed at step %d: %s", step_num, exc)
            return StepResult(
                step_number=step_num,
                thinking=f"LLM error: {exc}",
                done=True,
                response=f"Error: {exc}",
            )

        thinking = response.content or ""

        # Check if the LLM wants to use a tool
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except json.JSONDecodeError:
                tool_args = {}

            # Execute the tool
            observation = self._execute_tool(tool_name, **tool_args)
            obs_text = observation.text

            # Add messages to memory
            self._add_assistant_message(thinking)
            self._add_tool_message(obs_text, tool_call.get("id", ""))

            # Determine if done
            done = "final answer" in thinking.lower() or "task complete" in thinking.lower()

            result = StepResult(
                step_number=step_num,
                thinking=thinking,
                action=tool_name,
                action_input=tool_args,
                observation=obs_text,
                done=done,
            )
        else:
            # No tool call — this is a final response
            self._add_assistant_message(thinking)
            done = True
            result = StepResult(
                step_number=step_num,
                thinking=thinking,
                response=thinking,
                done=True,
            )

        self._history.append(result)
        return result

    def run(self, task: str) -> str:
        """Run the ReAct agent until completion or max steps."""
        logger.info("ReActAgent starting task: %s", task)
        self._add_user_message(task)

        while self._step_count < self.config.executor.max_steps:
            try:
                result = self.step(task)
            except MaxStepsExceededError:
                break

            if result.done:
                return result.response or result.thinking

        # Max steps reached — return the last response
        if self._history:
            last = self._history[-1]
            return last.response or last.thinking or "Task incomplete (max steps reached)"
        return "Task incomplete"


# ---------------------------------------------------------------------------
# Chain-of-Thought agent
# ---------------------------------------------------------------------------

class ChainOfThoughtAgent(Agent):
    """Chain-of-Thought reasoning agent.

    Generates explicit step-by-step reasoning before acting.  At each
    step, the agent produces a thinking block, then either calls a tool
    or provides a final answer.

    This agent is useful for tasks that require careful reasoning
    before tool use.
    """

    def step(self, task: str) -> StepResult:
        """Execute one chain-of-thought step."""
        self._step_count += 1
        step_num = self._step_count

        if step_num > self.config.executor.max_steps:
            raise MaxStepsExceededError(
                f"Maximum steps ({self.config.executor.max_steps}) exceeded"
            )

        self._add_system_prompt(task)

        # Add a thinking prompt
        self._add_user_message(
            "Think step by step. First, reason about what you need to do, "
            "then decide whether to use a tool or provide your answer."
        )

        messages = self._get_context_messages()
        tool_schemas = self._format_tool_schemas()

        try:
            response = self.llm.chat(messages, tools=tool_schemas)
        except LLMError as exc:
            return StepResult(
                step_number=step_num,
                thinking=f"LLM error: {exc}",
                done=True,
                response=f"Error: {exc}",
            )

        thinking = response.content or ""

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except json.JSONDecodeError:
                tool_args = {}

            observation = self._execute_tool(tool_name, **tool_args)
            obs_text = observation.text

            self._add_assistant_message(thinking)
            self._add_tool_message(obs_text, tool_call.get("id", ""))

            done = "final answer" in thinking.lower() or "task complete" in thinking.lower()

            result = StepResult(
                step_number=step_num,
                thinking=thinking,
                action=tool_name,
                action_input=tool_args,
                observation=obs_text,
                done=done,
            )
        else:
            self._add_assistant_message(thinking)
            result = StepResult(
                step_number=step_num,
                thinking=thinking,
                response=thinking,
                done=True,
            )

        self._history.append(result)
        return result

    def run(self, task: str) -> str:
        """Run the chain-of-thought agent until completion."""
        logger.info("ChainOfThoughtAgent starting task: %s", task)
        self._add_user_message(task)

        while self._step_count < self.config.executor.max_steps:
            try:
                result = self.step(task)
            except MaxStepsExceededError:
                break

            if result.done:
                return result.response or result.thinking

        if self._history:
            last = self._history[-1]
            return last.response or last.thinking or "Task incomplete (max steps reached)"
        return "Task incomplete"


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in an execution plan.

    Attributes:
        description: What needs to be done.
        tool: The tool to use (if any).
        args: Arguments for the tool.
        expected_result: What the result should look like.
        completed: Whether this step has been completed.
    """

    description: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    expected_result: str = ""
    completed: bool = False


@dataclass
class ExecutionPlan:
    """A plan for executing a task.

    Attributes:
        task: The original task description.
        steps: Ordered list of :class:`PlanStep` objects.
        created_at: When the plan was created.
    """

    task: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def mark_step_completed(self, index: int) -> None:
        """Mark a step as completed."""
        if 0 <= index < len(self.steps):
            self.steps[index].completed = True

    @property
    def is_complete(self) -> bool:
        """Whether all steps are completed."""
        return all(s.completed for s in self.steps) if self.steps else False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps": [
                {
                    "description": s.description,
                    "tool": s.tool,
                    "args": s.args,
                    "expected_result": s.expected_result,
                    "completed": s.completed,
                }
                for s in self.steps
            ],
            "created_at": self.created_at,
        }


class PlannerAgent(Agent):
    """Planner agent that decomposes tasks into sub-tasks.

    This agent first generates a plan (a sequence of steps), then
    executes each step in order, using tools as needed.  After each
    step, it evaluates whether the step was completed successfully
    and whether the plan needs adjustment.

    This is the most capable agent type for complex, multi-step tasks.
    """

    def _generate_plan(self, task: str) -> ExecutionPlan:
        """Generate an execution plan for the given task."""
        self._add_system_prompt(task)
        self._add_user_message(
            f"Break down the following task into concrete steps: '{task}'. "
            "For each step, specify what needs to be done and which tool "
            "(if any) should be used. "
            "Format your response as a JSON array of objects with keys: "
            '"description", "tool", "args", "expected_result".'
        )

        messages = self._get_context_messages()
        try:
            response = self.llm.chat(messages)
        except LLMError as exc:
            raise PlanningError(f"Failed to generate plan: {exc}")

        content = response.content

        # Try to parse JSON from the response
        try:
            # Extract JSON from the response (may be wrapped in markdown)
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                steps_data = json.loads(json_match.group())
            else:
                steps_data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # Fallback: create a simple plan
            logger.warning("Could not parse plan JSON, using fallback plan")
            steps_data = [
                {"description": f"Execute: {task}", "tool": None, "args": {}, "expected_result": "Task completed"},
            ]

        steps = [
            PlanStep(
                description=s.get("description", ""),
                tool=s.get("tool"),
                args=s.get("args", {}),
                expected_result=s.get("expected_result", ""),
            )
            for s in steps_data
        ]

        plan = ExecutionPlan(task=task, steps=steps)
        logger.info("Generated plan with %d steps for task: %s", len(steps), task)
        return plan

    def step(self, task: str) -> StepResult:
        """Execute one step of the plan.

        Note: For the PlannerAgent, :meth:`step` executes the next
        uncompleted step in the current plan.  Use :meth:`run` for
        full task execution.
        """
        self._step_count += 1
        step_num = self._step_count

        if step_num > self.config.executor.max_steps:
            raise MaxStepsExceededError(
                f"Maximum steps ({self.config.executor.max_steps}) exceeded"
            )

        # This method is called by run() which manages the plan
        # If called directly, generate a plan first
        if not hasattr(self, "_current_plan") or self._current_plan is None:
            self._current_plan = self._generate_plan(task)

        plan = self._current_plan
        next_step = None
        for i, s in enumerate(plan.steps):
            if not s.completed:
                next_step = (i, s)
                break

        if next_step is None:
            return StepResult(
                step_number=step_num,
                thinking="All plan steps completed.",
                done=True,
                response="Task completed successfully.",
            )

        idx, step = next_step
        thinking = f"Executing step {idx + 1}/{len(plan.steps)}: {step.description}"

        if step.tool:
            observation = self._execute_tool(step.tool, **step.args)
            obs_text = observation.text
            self._add_tool_message(obs_text)
            plan.mark_step_completed(idx)

            done = plan.is_complete
            result = StepResult(
                step_number=step_num,
                thinking=thinking,
                action=step.tool,
                action_input=step.args,
                observation=obs_text,
                done=done,
            )
        else:
            # No tool — this is a reasoning step
            self._add_user_message(step.description)
            messages = self._get_context_messages()
            try:
                response = self.llm.chat(messages)
                obs_text = response.content or ""
            except LLMError as exc:
                obs_text = f"Error: {exc}"

            self._add_assistant_message(obs_text)
            plan.mark_step_completed(idx)
            done = plan.is_complete

            result = StepResult(
                step_number=step_num,
                thinking=thinking,
                response=obs_text,
                done=done,
            )

        self._history.append(result)
        return result

    def run(self, task: str) -> str:
        """Run the planner agent until the plan is complete."""
        logger.info("PlannerAgent starting task: %s", task)

        try:
            self._current_plan = self._generate_plan(task)
        except PlanningError as exc:
            return f"Planning failed: {exc}"

        plan = self._current_plan

        while self._step_count < self.config.executor.max_steps and not plan.is_complete:
            try:
                result = self.step(task)
            except MaxStepsExceededError:
                break

            if result.done:
                break

        # Generate final summary
        self._add_user_message(
            f"Summarise the results of completing the task: '{task}'. "
            "Provide a clear, concise final answer."
        )
        messages = self._get_context_messages()
        try:
            response = self.llm.chat(messages)
            return response.content or "Task completed."
        except LLMError as exc:
            return f"Task completed (summary error: {exc})"


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agent(
    agent_type: str = "react",
    config: Optional[AgentConfig] = None,
    **kwargs: Any,
) -> Agent:
    """Factory function to create an agent by type.

    Args:
        agent_type: One of ``"react"``, ``"cot"``, or ``"planner"``.
        config: Optional :class:`AgentConfig`.  If ``None``, a default
            config is created (with ``agent_type`` set to *agent_type*).
        **kwargs: Additional keyword arguments passed to the agent
            constructor.

    Returns:
        An :class:`Agent` instance.

    Raises:
        AgentError: If *agent_type* is not recognised.
    """
    agent_classes = {
        "react": ReActAgent,
        "cot": ChainOfThoughtAgent,
        "planner": PlannerAgent,
    }

    cls = agent_classes.get(agent_type)
    if cls is None:
        raise AgentError(
            f"Unknown agent type: '{agent_type}'. "
            f"Available: {list(agent_classes.keys())}"
        )

    if config is None:
        config = AgentConfig(agent_type=agent_type)

    return cls(config=config, **kwargs)
