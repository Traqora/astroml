"""Autonomous task execution engine.

The :class:`TaskExecutor` and :class:`AutonomousExecutor` provide a
high-level interface for running complex tasks autonomously.  They:

1. **Plan** the task into sub-steps (using an :class:`~astroml.agent.base.Agent`).
2. **Execute** each step, using tools and LLM calls as needed.
3. **Verify** the results against success criteria.
4. **Recover** from failures by retrying with a different approach.
5. **Report** the final outcome.

Example::

    from astroml.agent.executor import TaskExecutor, Task
    from astroml.agent.base import create_agent

    agent = create_agent(agent_type="planner")
    executor = TaskExecutor(agent=agent)
    task = Task(description="Calculate the area of a circle with radius 5")
    result = executor.execute(task)
    print(result.success, result.output)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .base import Agent, ExecutionPlan, PlanStep, StepResult, create_agent
from .config import AgentConfig, ExecutorConfig
from .exceptions import (
    AgentError,
    MaxStepsExceededError,
    PlanningError,
    TaskFailedError,
    TaskTimeoutError,
)
from .memory import Episode
from .tools import ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task data structure
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A task to be executed autonomously.

    Attributes:
        description: What needs to be done.
        success_criteria: Optional list of conditions that must be met
            for the task to be considered successful.
        constraints: Optional list of constraints (e.g. "use only built-in tools").
        context: Optional additional context or background information.
        timeout: Optional per-task timeout in seconds.
        metadata: Optional dict of extra information.
    """

    description: str
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    context: str = ""
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Format the task as a prompt string for the agent."""
        parts = [f"Task: {self.description}"]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.success_criteria:
            parts.append("Success criteria:")
            for i, crit in enumerate(self.success_criteria, 1):
                parts.append(f"  {i}. {crit}")
        if self.constraints:
            parts.append("Constraints:")
            for i, con in enumerate(self.constraints, 1):
                parts.append(f"  {i}. {con}")
        return "\n".join(parts)


@dataclass
class TaskResult:
    """Result of executing a task.

    Attributes:
        success: Whether the task was completed successfully.
        output: The final output or summary.
        task: The original task description.
        steps: List of step results.
        plan: The execution plan used (if any).
        elapsed_seconds: Total time taken.
        error: Error message if the task failed.
        metadata: Optional dict of extra information.
    """

    success: bool
    output: str
    task: str
    steps: List[StepResult] = field(default_factory=list)
    plan: Optional[ExecutionPlan] = None
    elapsed_seconds: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "plan": self.plan.to_dict() if self.plan else None,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Task executor
# ---------------------------------------------------------------------------

class TaskExecutor:
    """Executes tasks using an agent with planning and recovery.

    This is the primary interface for autonomous task execution.  It
    wraps an :class:`~astroml.agent.base.Agent` and adds:

    * Task timeout enforcement.
    * Step-by-step execution with progress tracking.
    * Failure recovery with retry.
    * Episode recording for learning.

    Example::

        executor = TaskExecutor(agent=create_agent("planner"))
        result = executor.execute(Task(description="List files in current directory"))
        assert result.success
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.agent = agent or create_agent(agent_type="planner")
        self.config = config or ExecutorConfig()
        self._current_plan: Optional[ExecutionPlan] = None

    def execute(self, task: Task) -> TaskResult:
        """Execute a task autonomously.

        Args:
            task: The :class:`Task` to execute.

        Returns:
            A :class:`TaskResult` with the outcome.
        """
        start_time = time.time()
        logger.info("TaskExecutor executing: %s", task.description)

        # Reset agent state
        self.agent.reset()

        # Build the prompt
        prompt = task.to_prompt()
        if task.context:
            prompt = f"Context: {task.context}\n\n{prompt}"

        # Execute with timeout
        timeout = task.timeout or self.config.task_timeout
        try:
            output = self._execute_with_timeout(prompt, timeout)
        except TaskTimeoutError as exc:
            elapsed = time.time() - start_time
            return TaskResult(
                success=False,
                output="",
                task=task.description,
                steps=self.agent.get_history(),
                elapsed_seconds=elapsed,
                error=str(exc),
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("Task execution failed: %s", exc, exc_info=True)
            return TaskResult(
                success=False,
                output="",
                task=task.description,
                steps=self.agent.get_history(),
                elapsed_seconds=elapsed,
                error=str(exc),
            )

        elapsed = time.time() - start_time

        # Record episode
        episode = Episode(
            task=task.description,
            steps=[s.to_dict() for s in self.agent.get_history()],
            success=True,
            result=output,
            started_at=start_time,
            completed_at=time.time(),
            metadata={"elapsed_seconds": elapsed},
        )
        self.agent.memory.add_episode(episode)

        return TaskResult(
            success=True,
            output=output,
            task=task.description,
            steps=self.agent.get_history(),
            plan=getattr(self.agent, "_current_plan", None),
            elapsed_seconds=elapsed,
        )

    def _execute_with_timeout(self, prompt: str, timeout: float) -> str:
        """Execute the agent with a timeout."""
        import threading

        result_container: Dict[str, Any] = {}

        def _run():
            try:
                result_container["output"] = self.agent.run(prompt)
                result_container["success"] = True
            except Exception as exc:
                result_container["error"] = exc
                result_container["success"] = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TaskTimeoutError(
                f"Task exceeded timeout of {timeout} seconds"
            )

        if not result_container.get("success"):
            raise result_container.get("error", AgentError("Unknown execution error"))

        return result_container.get("output", "")

    def execute_batch(
        self,
        tasks: Sequence[Task],
        parallel: bool = False,
    ) -> List[TaskResult]:
        """Execute multiple tasks.

        Args:
            tasks: Sequence of :class:`Task` objects.
            parallel: If ``True``, execute tasks in parallel using
                threads.  Note: parallel execution shares the same
                agent and memory, so results may interleave.

        Returns:
            List of :class:`TaskResult` objects, one per task.
        """
        if parallel:
            import concurrent.futures

            results: List[TaskResult] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_task = {
                    executor.submit(self.execute, task): task for task in tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(TaskResult(
                            success=False,
                            output="",
                            task=task.description,
                            error=str(exc),
                        ))
            return results
        else:
            return [self.execute(task) for task in tasks]

    def verify_success(self, result: TaskResult, task: Task) -> bool:
        """Verify that a task result meets the success criteria.

        Args:
            result: The :class:`TaskResult` to verify.
            task: The original :class:`Task`.

        Returns:
            ``True`` if all success criteria are met.
        """
        if not result.success:
            return False

        # If no explicit success criteria, just check the result is non-empty
        if not task.success_criteria:
            return bool(result.output.strip())

        # Check each criterion against the output
        output_lower = result.output.lower()
        for criterion in task.success_criteria:
            crit_lower = criterion.lower()
            # Simple keyword matching — for production, use an LLM judge
            if crit_lower not in output_lower:
                logger.warning("Success criterion not met: %s", criterion)
                return False

        return True


# ---------------------------------------------------------------------------
# Autonomous executor
# ---------------------------------------------------------------------------

class AutonomousExecutor:
    """High-level autonomous executor with recovery and reporting.

    Wraps :class:`TaskExecutor` and adds:

    * Automatic recovery from failures (retry with different agent type).
    * Progress reporting callbacks.
    * Result aggregation and summarisation.

    Example::

        executor = AutonomousExecutor()
        result = executor.run("Calculate the Fibonacci sequence up to 100")
        print(result.output)
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        agent_config: Optional[AgentConfig] = None,
    ) -> None:
        self.config = config or ExecutorConfig()
        self.agent_config = agent_config or AgentConfig()
        self._executor: Optional[TaskExecutor] = None

    def _get_executor(self, agent_type: str = "planner") -> TaskExecutor:
        """Get or create a :class:`TaskExecutor` with the given agent type."""
        if self._executor is None:
            agent = create_agent(
                agent_type=agent_type,
                config=self.agent_config,
            )
            self._executor = TaskExecutor(agent=agent, config=self.config)
        return self._executor

    def run(
        self,
        task_description: str,
        success_criteria: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        context: str = "",
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> TaskResult:
        """Run a task autonomously with recovery.

        Args:
            task_description: What needs to be done.
            success_criteria: Optional list of success criteria.
            constraints: Optional list of constraints.
            context: Optional background context.
            timeout: Optional timeout in seconds.
            on_progress: Optional callback ``(step, total, message)``
                called after each step.

        Returns:
            A :class:`TaskResult` with the outcome.
        """
        task = Task(
            description=task_description,
            success_criteria=success_criteria or [],
            constraints=constraints or [],
            context=context,
            timeout=timeout,
        )

        # Try with the configured agent type first
        agent_type = self.agent_config.agent_type
        result = self._attempt_execution(task, agent_type)

        # Recovery: if failed, try with a different agent type
        if not result.success and self.config.recovery_attempts > 0:
            fallback_types = ["react", "cot", "planner"]
            fallback_types = [t for t in fallback_types if t != agent_type]

            for attempt in range(self.config.recovery_attempts):
                if attempt >= len(fallback_types):
                    break
                fallback_type = fallback_types[attempt]
                logger.info("Recovery attempt %d with agent type: %s",
                            attempt + 1, fallback_type)

                # Reset executor for a fresh attempt
                self._executor = None
                result = self._attempt_execution(task, fallback_type)
                if result.success:
                    break

        # Report progress
        if on_progress:
            steps = result.steps
            for i, step in enumerate(steps, 1):
                msg = step.response or step.thinking or ""
                on_progress(i, len(steps), msg)

        return result

    def _attempt_execution(self, task: Task, agent_type: str) -> TaskResult:
        """Attempt to execute a task with a specific agent type."""
        executor = self._get_executor(agent_type)
        try:
            result = executor.execute(task)
            # Verify success criteria
            if executor.verify_success(result, task):
                return result
            else:
                # Criteria not met — mark as failed
                return TaskResult(
                    success=False,
                    output=result.output,
                    task=result.task,
                    steps=result.steps,
                    plan=result.plan,
                    elapsed_seconds=result.elapsed_seconds,
                    error="Success criteria not met",
                )
        except Exception as exc:
            logger.error("Execution attempt failed: %s", exc, exc_info=True)
            return TaskResult(
                success=False,
                output="",
                task=task.description,
                error=str(exc),
            )

    def run_batch(
        self,
        tasks: List[str],
        **kwargs: Any,
    ) -> List[TaskResult]:
        """Run multiple tasks autonomously.

        Args:
            tasks: List of task description strings.
            **kwargs: Additional arguments passed to :meth:`run`.

        Returns:
            List of :class:`TaskResult` objects.
        """
        results: List[TaskResult] = []
        for task_desc in tasks:
            result = self.run(task_desc, **kwargs)
            results.append(result)
        return results

    def summarize(self, results: List[TaskResult]) -> str:
        """Generate a summary of multiple task results.

        Args:
            results: List of :class:`TaskResult` objects.

        Returns:
            A summary string.
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        total_time = sum(r.elapsed_seconds for r in results)

        lines = [
            f"Autonomous Execution Summary",
            f"=" * 40,
            f"Total tasks:  {total}",
            f"Successful:   {successful}",
            f"Failed:       {failed}",
            f"Total time:   {total_time:.2f}s",
            "",
        ]

        for i, result in enumerate(results, 1):
            status = "✓" if result.success else "✗"
            lines.append(f"  {status} Task {i}: {result.task[:60]}")
            if result.success:
                preview = result.output[:100].replace("\n", " ")
                lines.append(f"      → {preview}")
            else:
                lines.append(f"      → Error: {result.error[:100]}")

        return "\n".join(lines)
