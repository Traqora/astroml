"""Core Agent implementation with multi-step reasoning and autonomous execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from .memory import MemoryStore
from .planning import Planner, Task, TaskPlan
from .tools import Tool, ToolRegistry


@dataclass
class AgentConfig:
    """Configuration for an Agent instance."""

    name: str = "astroml-agent"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_steps: int = 10
    max_tokens: int = 2000
    enable_planning: bool = True
    enable_memory: bool = True
    enable_reflection: bool = True
    verbose: bool = False


@dataclass
class AgentStep:
    """Represents a single step in agent execution."""

    step_number: int
    thought: str
    action: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResult:
    """Result of agent execution."""

    final_answer: str
    steps: List[AgentStep]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Multi-step reasoning agent with autonomous task execution capabilities."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Sequence[Tool]] = None,
        memory: Optional[MemoryStore] = None,
        planner: Optional[Planner] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        """Initialize the Agent.

        Args:
            config: Agent configuration
            tools: List of available tools
            memory: Memory store for context
            planner: Task planner
            llm_client: LLM client for inference
        """
        self.config = config or AgentConfig()
        self.tool_registry = ToolRegistry(tools or [])
        self.memory = memory or MemoryStore()
        self.planner = planner or Planner()
        self.llm_client = llm_client
        self.steps: List[AgentStep] = []

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Execute a task with multi-step reasoning.

        Args:
            task: The task description
            context: Additional context for the task

        Returns:
            AgentResult with final answer and execution trace
        """
        context = context or {}
        self.steps = []

        try:
            # Store initial task in memory
            if self.config.enable_memory:
                self.memory.add_message("user", task, context)

            # Plan the task if enabled
            if self.config.enable_planning:
                plan = self.planner.plan(task, context)
                if self.config.verbose:
                    print(f"Generated plan: {plan}")
            else:
                plan = None

            # Execute reasoning loop
            for step_num in range(1, self.config.max_steps + 1):
                step = self._execute_step(step_num, task, context, plan)
                self.steps.append(step)

                if self.config.verbose:
                    print(f"Step {step_num}: {step.thought}")
                    if step.action:
                        print(f"  Action: {step.action}")
                    if step.observation:
                        print(f"  Observation: {step.observation}")

                # Check if we have a final answer
                if self._is_final_answer(step):
                    final_answer = step.observation or step.thought
                    if self.config.enable_memory:
                        self.memory.add_message("assistant", final_answer, context)
                    return AgentResult(
                        final_answer=final_answer,
                        steps=self.steps,
                        success=True,
                    )

                # Update context with observation
                if step.observation:
                    context["last_observation"] = step.observation

            # Max steps reached without final answer
            return AgentResult(
                final_answer="Max steps reached without conclusion",
                steps=self.steps,
                success=False,
                error="Execution limit exceeded",
            )

        except Exception as e:
            return AgentResult(
                final_answer="",
                steps=self.steps,
                success=False,
                error=str(e),
            )

    def _execute_step(
        self,
        step_number: int,
        task: str,
        context: Dict[str, Any],
        plan: Optional[TaskPlan],
    ) -> AgentStep:
        """Execute a single reasoning step.

        Args:
            step_number: Current step number
            task: Original task
            context: Current context
            plan: Task plan if available

        Returns:
            AgentStep with thought, action, and observation
        """
        # Build prompt with context
        prompt = self._build_prompt(task, context, plan)

        # Get reasoning from LLM
        thought = self._get_llm_response(prompt)

        # Parse thought for action/tool call
        action, tool_call = self._parse_thought(thought)

        observation = None
        if tool_call:
            # Execute tool call
            observation = self._execute_tool_call(tool_call, context)

        return AgentStep(
            step_number=step_number,
            thought=thought,
            action=action,
            tool_call=tool_call,
            observation=observation,
        )

    def _build_prompt(
        self,
        task: str,
        context: Dict[str, Any],
        plan: Optional[TaskPlan],
    ) -> str:
        """Build prompt for LLM inference.

        Args:
            task: Current task
            context: Execution context
            plan: Task plan

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Task: {task}"]

        if plan:
            prompt_parts.append(f"\nPlan: {plan.description}")
            for i, subtask in enumerate(plan.tasks, 1):
                prompt_parts.append(f"  {i}. {subtask.description}")

        if self.config.enable_memory:
            history = self.memory.get_recent_messages(n=5)
            if history:
                prompt_parts.append("\nConversation History:")
                for msg in history:
                    prompt_parts.append(f"  {msg.role}: {msg.content}")

        if "last_observation" in context:
            prompt_parts.append(f"\nLast Observation: {context['last_observation']}")

        prompt_parts.append(f"\nAvailable Tools: {self.tool_registry.list_tools()}")

        prompt_parts.append(
            "\n\nThink step by step. If you need to use a tool, format your response as:\n"
            "THOUGHT: [your reasoning]\n"
            "ACTION: tool_name\n"
            "INPUT: [tool input as JSON]\n\n"
            "If you have the final answer, format as:\n"
            "THOUGHT: [your reasoning]\n"
            "FINAL_ANSWER: [your answer]"
        )

        return "\n".join(prompt_parts)

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM.

        Args:
            prompt: Input prompt

        Returns:
            LLM response string
        """
        if self.llm_client:
            # Use provided LLM client
            response = self.llm_client.generate(
                prompt=prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response
        else:
            # Mock response for testing without LLM
            return "THOUGHT: I need to analyze the task and determine the next action."

    def _parse_thought(self, thought: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse thought for action and tool call.

        Args:
            thought: Raw thought string

        Returns:
            Tuple of (action, tool_call_dict)
        """
        lines = thought.split("\n")
        action = None
        tool_call = None

        for i, line in enumerate(lines):
            if line.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip()
                # Look for INPUT on next line
                if i + 1 < len(lines) and lines[i + 1].startswith("INPUT:"):
                    try:
                        input_json = lines[i + 1].split(":", 1)[1].strip()
                        tool_call = {"tool": action, "input": json.loads(input_json)}
                    except json.JSONDecodeError:
                        tool_call = {"tool": action, "input": {}}
                else:
                    tool_call = {"tool": action, "input": {}}
            elif line.startswith("FINAL_ANSWER:"):
                action = "final_answer"
                break

        return action, tool_call

    def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Execute a tool call.

        Args:
            tool_call: Tool call dictionary
            context: Execution context

        Returns:
            Tool observation/result
        """
        tool_name = tool_call["tool"]
        tool_input = tool_call.get("input", {})

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        try:
            result = tool.execute(tool_input, context)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _is_final_answer(self, step: AgentStep) -> bool:
        """Check if step contains a final answer.

        Args:
            step: Agent step to check

        Returns:
            True if step has final answer
        """
        return step.action == "final_answer" or (
            step.observation and "FINAL_ANSWER:" in step.observation
        )

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's registry.

        Args:
            tool: Tool to add
        """
        self.tool_registry.register(tool)

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get execution trace as list of dictionaries.

        Returns:
            List of step dictionaries
        """
        return [
            {
                "step": step.step_number,
                "thought": step.thought,
                "action": step.action,
                "tool_call": step.tool_call,
                "observation": step.observation,
                "timestamp": step.timestamp.isoformat(),
            }
            for step in self.steps
        ]
