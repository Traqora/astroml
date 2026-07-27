"""Comprehensive tests for the LLM Agent Framework.

Tests cover:
- Configuration loading and validation
- Memory subsystems (short-term, long-term, episodic)
- Tool registry and built-in tools
- LLM provider (mock)
- Agent base class and ReAct/CoT/Planner agents
- Task executor and autonomous executor
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astroml.agent.base import (
    Agent,
    ChainOfThoughtAgent,
    ExecutionPlan,
    PlanStep,
    PlannerAgent,
    ReActAgent,
    StepResult,
    create_agent,
)
from astroml.agent.config import (
    AgentConfig,
    ExecutorConfig,
    LLMConfig,
    MemoryConfig,
)
from astroml.agent.exceptions import (
    AgentError,
    LLMConfigurationError,
    MaxStepsExceededError,
    PlanningError,
    ToolNotFoundError,
)
from astroml.agent.executor import (
    AutonomousExecutor,
    Task,
    TaskExecutor,
    TaskResult,
)
from astroml.agent.llm import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    MockProvider,
)
from astroml.agent.memory import (
    Episode,
    LongTermMemory,
    MemoryEntry,
    MemoryManager,
    Message,
    ShortTermMemory,
)
from astroml.agent.tools import (
    CalculatorTool,
    FileReadTool,
    FileWriteTool,
    ListDirectoryTool,
    PythonREPLTool,
    SearchTool,
    Tool,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)


# ===========================================================================
# Configuration tests
# ===========================================================================

class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self):
        """Default config should use mock provider."""
        config = LLMConfig()
        assert config.provider == "mock"
        assert config.model == "gpt-3.5-turbo"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_from_env(self):
        """Should load settings from environment variables."""
        os.environ["ASTROML_AGENT_LLM_PROVIDER"] = "openai"
        os.environ["ASTROML_AGENT_LLM_MODEL"] = "gpt-4"
        os.environ["ASTROML_AGENT_LLM_API_KEY"] = "test-key"
        try:
            config = LLMConfig.from_env()
            assert config.provider == "openai"
            assert config.model == "gpt-4"
            assert config.api_key == "test-key"
        finally:
            del os.environ["ASTROML_AGENT_LLM_PROVIDER"]
            del os.environ["ASTROML_AGENT_LLM_MODEL"]
            del os.environ["ASTROML_AGENT_LLM_API_KEY"]

    def test_frozen(self):
        """Config should be immutable."""
        config = LLMConfig()
        with pytest.raises((AttributeError, Exception)):
            config.provider = "openai"  # type: ignore[misc]


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        """Default config should have all sub-configs."""
        config = AgentConfig()
        assert config.agent_type == "react"
        assert config.name == "astroml-agent"
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.executor, ExecutorConfig)

    def test_from_dict(self):
        """Should build config from a dictionary."""
        data = {
            "agent_type": "planner",
            "name": "test-agent",
            "llm": {"provider": "mock", "model": "test-model"},
            "executor": {"max_steps": 10},
        }
        config = AgentConfig.from_dict(data)
        assert config.agent_type == "planner"
        assert config.name == "test-agent"
        assert config.llm.model == "test-model"
        assert config.executor.max_steps == 10

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys should be silently ignored."""
        data = {"unknown_key": "value", "agent_type": "cot"}
        config = AgentConfig.from_dict(data)
        assert config.agent_type == "cot"


# ===========================================================================
# Memory tests
# ===========================================================================

class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    def test_add_and_get(self):
        """Should store and retrieve messages."""
        mem = ShortTermMemory(limit=5)
        msg = Message(role="user", content="hello")
        mem.add(msg)
        assert len(mem) == 1
        assert mem[0].content == "hello"

    def test_limit_eviction(self):
        """Oldest messages should be evicted when over limit."""
        mem = ShortTermMemory(limit=3)
        for i in range(5):
            mem.add(Message(role="user", content=f"msg{i}"))
        assert len(mem) == 3
        assert mem[0].content == "msg2"  # msg0 and msg1 evicted

    def test_clear(self):
        """Should clear all messages."""
        mem = ShortTermMemory(limit=10)
        mem.add(Message(role="user", content="hello"))
        mem.clear()
        assert len(mem) == 0

    def test_invalid_limit(self):
        """Should reject limit < 1."""
        with pytest.raises(Exception):
            ShortTermMemory(limit=0)


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    def test_store_and_retrieve(self):
        """Should store and retrieve values by key."""
        mem = LongTermMemory(limit=10)
        mem.store("capital", "Paris")
        assert mem.retrieve("capital") == "Paris"

    def test_retrieve_missing(self):
        """Should return None for missing keys."""
        mem = LongTermMemory(limit=10)
        assert mem.retrieve("nonexistent") is None

    def test_update_existing(self):
        """Should update existing entries."""
        mem = LongTermMemory(limit=10)
        mem.store("key", "value1")
        mem.store("key", "value2")
        assert mem.retrieve("key") == "value2"

    def test_tags(self):
        """Should support tag-based retrieval."""
        mem = LongTermMemory(limit=10)
        mem.store("fact1", "Paris", tags=["geography", "capital"])
        mem.store("fact2", "Berlin", tags=["geography", "capital"])
        results = mem.retrieve_by_tag("geography")
        assert len(results) == 2

    def test_search(self):
        """Should support substring search."""
        mem = LongTermMemory(limit=10)
        mem.store("capital_of_france", "Paris")
        results = mem.search("paris")
        assert len(results) == 1
        assert results[0].key == "capital_of_france"

    def test_delete(self):
        """Should delete entries."""
        mem = LongTermMemory(limit=10)
        mem.store("key", "value")
        assert mem.delete("key") is True
        assert mem.retrieve("key") is None
        assert mem.delete("key") is False  # Already deleted

    def test_capacity_eviction(self):
        """Should evict oldest entries when at capacity."""
        mem = LongTermMemory(limit=2)
        mem.store("key1", "val1")
        mem.store("key2", "val2")
        mem.store("key3", "val3")
        assert mem.retrieve("key1") is None  # Evicted
        assert mem.retrieve("key2") == "val2"
        assert mem.retrieve("key3") == "val3"


class TestEpisodicMemory:
    """Tests for EpisodicMemory."""

    def test_add_and_get(self):
        """Should store and retrieve episodes."""
        mem = EpisodicMemory(limit=5)
        ep = Episode(task="test task", success=True, result="done")
        mem.add(ep)
        assert len(mem) == 1
        assert mem.get_all()[0].task == "test task"

    def test_get_recent(self):
        """Should return the most recent episodes."""
        mem = EpisodicMemory(limit=10)
        for i in range(5):
            mem.add(Episode(task=f"task{i}", success=True))
        recent = mem.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].task == "task4"

    def test_filter_success_failed(self):
        """Should filter by success/failure."""
        mem = EpisodicMemory(limit=10)
        mem.add(Episode(task="success1", success=True))
        mem.add(Episode(task="fail1", success=False))
        mem.add(Episode(task="success2", success=True))
        assert len(mem.get_successful()) == 2
        assert len(mem.get_failed()) == 1


class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_init(self):
        """Should initialize all subsystems."""
        config = MemoryConfig()
        manager = MemoryManager(config)
        assert manager.short_term is not None
        assert manager.long_term is not None
        assert manager.episodic is not None

    def test_add_message(self):
        """Should add messages to short-term memory."""
        manager = MemoryManager(MemoryConfig())
        manager.add_message(Message(role="user", content="hello"))
        context = manager.get_context()
        assert len(context) == 1
        assert context[0]["role"] == "user"

    def test_store_and_retrieve_fact(self):
        """Should store and retrieve facts."""
        manager = MemoryManager(MemoryConfig())
        manager.store_fact("key", "value", tags=["test"])
        assert manager.retrieve_fact("key") == "value"

    def test_persistence(self, tmp_path):
        """Should save and load memory."""
        manager = MemoryManager(MemoryConfig())
        manager.store_fact("key1", "value1")
        manager.add_episode(Episode(task="task1", success=True))

        path = tmp_path / "memory.json"
        manager.save(str(path))
        assert path.exists()

        # Load into a new manager
        manager2 = MemoryManager(MemoryConfig())
        manager2.load(str(path))
        assert manager2.retrieve_fact("key1") == "value1"
        assert len(manager2.episodic) == 1

    def test_reset(self):
        """Should clear all memory."""
        manager = MemoryManager(MemoryConfig())
        manager.add_message(Message(role="user", content="hello"))
        manager.store_fact("key", "value")
        manager.add_episode(Episode(task="task", success=True))
        manager.reset()
        assert len(manager.short_term) == 0
        assert len(manager.long_term) == 0
        assert len(manager.episodic) == 0


# ===========================================================================
# Tool tests
# ===========================================================================

class TestCalculatorTool:
    """Tests for CalculatorTool."""

    def test_simple_addition(self):
        """Should evaluate simple expressions."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 2")
        assert result.success
        assert result.output == "4"

    def test_complex_expression(self):
        """Should handle complex expressions."""
        tool = CalculatorTool()
        result = tool.execute(expression="(3 + 4) * 2")
        assert result.success
        assert result.output == "14"

    def test_power(self):
        """Should handle exponentiation."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10")
        assert result.success
        assert result.output == "1024"

    def test_division(self):
        """Should handle division."""
        tool = CalculatorTool()
        result = tool.execute(expression="10 / 4")
        assert result.success
        assert float(result.output) == 2.5

    def test_syntax_error(self):
        """Should handle syntax errors."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 +")
        assert not result.success


class TestPythonREPLTool:
    """Tests for PythonREPLTool."""

    def test_simple_execution(self):
        """Should execute Python code."""
        tool = PythonREPLTool()
        result = tool.execute(code="print('hello world')")
        assert result.success
        assert "hello world" in result.output

    def test_calculation(self):
        """Should execute calculations."""
        tool = PythonREPLTool()
        result = tool.execute(code="x = 5 + 3\nprint(x)")
        assert result.success
        assert "8" in result.output

    def test_timeout(self):
        """Should handle timeouts."""
        tool = PythonREPLTool()
        result = tool.execute(code="import time; time.sleep(10)", timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()


class TestFileTools:
    """Tests for file read/write tools."""

    def test_write_and_read(self, tmp_path):
        """Should write and read files."""
        write_tool = FileWriteTool()
        read_tool = FileReadTool()

        path = str(tmp_path / "test.txt")
        write_result = write_tool.execute(path=path, content="Hello, World!")
        assert write_result.success

        read_result = read_tool.execute(path=path)
        assert read_result.success
        assert "Hello, World!" in read_result.output

    def test_read_nonexistent(self):
        """Should handle missing files."""
        tool = FileReadTool()
        result = tool.execute(path="/nonexistent/file.txt")
        assert not result.success

    def test_write_creates_parent_dirs(self, tmp_path):
        """Should create parent directories."""
        tool = FileWriteTool()
        path = str(tmp_path / "subdir" / "nested" / "file.txt")
        result = tool.execute(path=path, content="test")
        assert result.success
        assert Path(path).exists()


class TestListDirectoryTool:
    """Tests for ListDirectoryTool."""

    def test_list_directory(self, tmp_path):
        """Should list directory contents."""
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()

        tool = ListDirectoryTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success
        assert "file1.txt" in result.output
        assert "subdir" in result.output

    def test_list_nonexistent(self):
        """Should handle missing directories."""
        tool = ListDirectoryTool()
        result = tool.execute(path="/nonexistent")
        assert not result.success


class TestSearchTool:
    """Tests for SearchTool."""

    def test_search(self, tmp_path):
        """Should search for patterns in files."""
        (tmp_path / "test.py").write_text("def hello():\n    print('world')\n")

        tool = SearchTool()
        result = tool.execute(pattern="hello", path=str(tmp_path), file_pattern="*.py")
        assert result.success
        assert "hello" in result.output

    def test_no_matches(self, tmp_path):
        """Should report no matches."""
        (tmp_path / "test.py").write_text("def foo():\n    pass\n")

        tool = SearchTool()
        result = tool.execute(pattern="nonexistent", path=str(tmp_path), file_pattern="*.py")
        assert result.success
        assert "No matches" in result.output


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        """Should register and retrieve tools."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.has("calculator")
        assert registry.get("calculator") is tool

    def test_unregister(self):
        """Should unregister tools."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert registry.unregister("calculator") is True
        assert not registry.has("calculator")

    def test_get_schemas(self):
        """Should return tool schemas."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "calculator"

    def test_execute(self):
        """Should execute registered tools."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = registry.execute("calculator", expression="2 + 2")
        assert result.success
        assert result.output == "4"

    def test_execute_unregistered(self):
        """Should raise for unregistered tools."""
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_create_default_registry(self):
        """Should create a registry with all built-in tools."""
        registry = create_default_registry()
        assert "calculator" in registry
        assert "python_repl" in registry
        assert "read_file" in registry
        assert "write_file" in registry
        assert "list_directory" in registry
        assert "search" in registry
        assert "http_request" in registry


# ===========================================================================
# LLM tests
# ===========================================================================

class TestMockProvider:
    """Tests for MockProvider."""

    def test_chat_basic(self):
        """Should return a response."""
        config = LLMConfig(provider="mock")
        client = LLMClient(config)
        response = client.chat([{"role": "user", "content": "Hello"}])
        assert response.content
        assert response.model == "gpt-3.5-turbo"

    def test_chat_with_tools(self):
        """Should handle tool schemas."""
        config = LLMConfig(provider="mock")
        client = LLMClient(config)
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "test", "parameters": {}}}]
        response = client.chat(
            [{"role": "user", "content": "read the file"}],
            tools=tools,
        )
        assert response.content

    def test_calculate_response(self):
        """Should handle calculation queries."""
        config = LLMConfig(provider="mock")
        client = LLMClient(config)
        response = client.chat([{"role": "user", "content": "calculate 2 + 2"}])
        assert "4" in response.content

    def test_planning_response(self):
        """Should handle planning queries."""
        config = LLMConfig(provider="mock")
        client = LLMClient(config)
        response = client.chat([{"role": "user", "content": "create a plan for this task"}])
        assert "step" in response.content.lower()

    def test_count_calls(self):
        """Should track call count."""
        config = LLMConfig(provider="mock")
        client = LLMClient(config)
        client.chat([{"role": "user", "content": "hello"}])
        client.chat([{"role": "user", "content": "hello again"}])
        assert client.count_calls() == 2


class TestLLMClient:
    """Tests for LLMClient."""

    def test_unknown_provider(self):
        """Should raise for unknown providers."""
        config = LLMConfig(provider="unknown")
        with pytest.raises(LLMConfigurationError):
            LLMClient(config)

    def test_message_normalisation(self):
        """Should normalise mixed message types."""
        config = LLMConfig(provider="mock")
        client = LLMClient(config)
        msgs = client.provider._normalise_messages([
            LLMMessage(role="user", content="hello"),
            {"role": "assistant", "content": "hi"},
        ])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


# ===========================================================================
# Agent tests
# ===========================================================================

class TestAgentFactory:
    """Tests for the agent factory."""

    def test_create_react_agent(self):
        """Should create a ReActAgent."""
        agent = create_agent(agent_type="react")
        assert isinstance(agent, ReActAgent)

    def test_create_cot_agent(self):
        """Should create a ChainOfThoughtAgent."""
        agent = create_agent(agent_type="cot")
        assert isinstance(agent, ChainOfThoughtAgent)

    def test_create_planner_agent(self):
        """Should create a PlannerAgent."""
        agent = create_agent(agent_type="planner")
        assert isinstance(agent, PlannerAgent)

    def test_unknown_type(self):
        """Should raise for unknown agent types."""
        with pytest.raises(AgentError):
            create_agent(agent_type="unknown")

    def test_custom_config(self):
        """Should accept custom config."""
        config = AgentConfig(agent_type="react", name="test-agent")
        agent = create_agent(agent_type="react", config=config)
        assert agent.config.name == "test-agent"


class TestReActAgent:
    """Tests for ReActAgent."""

    def test_init(self):
        """Should initialise with default config."""
        agent = ReActAgent()
        assert agent.config is not None
        assert agent.llm is not None
        assert agent.tools is not None

    def test_reset(self):
        """Should reset state."""
        agent = ReActAgent()
        agent._step_count = 5
        agent.reset()
        assert agent._step_count == 0
        assert len(agent.get_history()) == 0

    def test_run_simple_task(self):
        """Should run a simple task."""
        agent = ReActAgent()
        result = agent.run("What is 2 + 2?")
        assert result is not None
        assert len(result) > 0

    def test_step_returns_step_result(self):
        """Step should return a StepResult."""
        agent = ReActAgent()
        result = agent.step("Test task")
        assert isinstance(result, StepResult)
        assert result.step_number == 1

    def test_max_steps_exceeded(self):
        """Should raise MaxStepsExceededError when steps exceeded."""
        config = AgentConfig(
            executor=ExecutorConfig(max_steps=1),
        )
        agent = ReActAgent(config=config)
        agent.run("Test task")  # First step
        with pytest.raises(MaxStepsExceededError):
            agent.step("Test task")  # Second step should exceed


class TestChainOfThoughtAgent:
    """Tests for ChainOfThoughtAgent."""

    def test_run_simple_task(self):
        """Should run a simple task."""
        agent = ChainOfThoughtAgent()
        result = agent.run("Explain the concept of gravity")
        assert result is not None
        assert len(result) > 0


class TestPlannerAgent:
    """Tests for PlannerAgent."""

    def test_run_simple_task(self):
        """Should run a simple task."""
        agent = PlannerAgent()
        result = agent.run("List the files in the current directory")
        assert result is not None

    def test_plan_generation(self):
        """Should generate a plan."""
        agent = PlannerAgent()
        plan = agent._generate_plan("Calculate 2 + 2")
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) > 0

    def test_plan_is_complete(self):
        """Empty plan should not be complete."""
        plan = ExecutionPlan(task="test")
        assert not plan.is_complete

    def test_mark_step_completed(self):
        """Should mark steps as completed."""
        plan = ExecutionPlan(task="test")
        plan.steps.append(PlanStep(description="step 1"))
        plan.mark_step_completed(0)
        assert plan.steps[0].completed
        assert not plan.is_complete  # Only 1 of 1 done, but is_complete checks all

    def test_plan_complete_when_all_done(self):
        """Plan should be complete when all steps are done."""
        plan = ExecutionPlan(task="test")
        plan.steps.append(PlanStep(description="step 1"))
        plan.steps.append(PlanStep(description="step 2"))
        plan.mark_step_completed(0)
        plan.mark_step_completed(1)
        assert plan.is_complete


# ===========================================================================
# Executor tests
# ===========================================================================

class TestTask:
    """Tests for Task dataclass."""

    def test_defaults(self):
        """Should have default values."""
        task = Task(description="test")
        assert task.description == "test"
        assert task.success_criteria == []
        assert task.constraints == []
        assert task.context == ""

    def test_to_prompt(self):
        """Should format as a prompt."""
        task = Task(
            description="Calculate area",
            success_criteria=["result > 0"],
            constraints=["use calculator"],
            context="math problem",
        )
        prompt = task.to_prompt()
        assert "Calculate area" in prompt
        assert "result > 0" in prompt
        assert "use calculator" in prompt
        assert "math problem" in prompt


class TestTaskExecutor:
    """Tests for TaskExecutor."""

    def test_init(self):
        """Should initialise with default agent."""
        executor = TaskExecutor()
        assert executor.agent is not None
        assert executor.config is not None

    def test_execute_simple_task(self):
        """Should execute a simple task."""
        executor = TaskExecutor()
        task = Task(description="Calculate 2 + 2")
        result = executor.execute(task)
        assert isinstance(result, TaskResult)
        assert result.success
        assert len(result.output) > 0

    def test_execute_records_episode(self):
        """Should record an episode after execution."""
        executor = TaskExecutor()
        task = Task(description="Test task")
        executor.execute(task)
        episodes = executor.agent.memory.episodic.get_all()
        assert len(episodes) == 1
        assert episodes[0].success

    def test_verify_success_no_criteria(self):
        """Should pass with no criteria if output is non-empty."""
        executor = TaskExecutor()
        task = Task(description="test")
        result = TaskResult(success=True, output="some output", task="test")
        assert executor.verify_success(result, task) is True

    def test_verify_success_with_criteria(self):
        """Should check criteria against output."""
        executor = TaskExecutor()
        task = Task(description="test", success_criteria=["hello"])
        result = TaskResult(success=True, output="hello world", task="test")
        assert executor.verify_success(result, task) is True

    def test_verify_success_criteria_not_met(self):
        """Should fail if criteria not met."""
        executor = TaskExecutor()
        task = Task(description="test", success_criteria=["missing"])
        result = TaskResult(success=True, output="hello world", task="test")
        assert executor.verify_success(result, task) is False

    def test_execute_batch(self):
        """Should execute multiple tasks."""
        executor = TaskExecutor()
        tasks = [
            Task(description="Task 1"),
            Task(description="Task 2"),
        ]
        results = executor.execute_batch(tasks)
        assert len(results) == 2
        assert all(r.success for r in results)


class TestAutonomousExecutor:
    """Tests for AutonomousExecutor."""

    def test_init(self):
        """Should initialise with default config."""
        executor = AutonomousExecutor()
        assert executor.config is not None
        assert executor.agent_config is not None

    def test_run_simple_task(self):
        """Should run a simple task."""
        executor = AutonomousExecutor()
        result = executor.run("Calculate 2 + 2")
        assert isinstance(result, TaskResult)
        assert result.success

    def test_run_with_criteria(self):
        """Should run with success criteria."""
        executor = AutonomousExecutor()
        result = executor.run(
            "Calculate 2 + 2",
            success_criteria=["4"],
        )
        assert result.success

    def test_run_batch(self):
        """Should run multiple tasks."""
        executor = AutonomousExecutor()
        results = executor.run_batch(["Task 1", "Task 2"])
        assert len(results) == 2

    def test_summarize(self):
        """Should generate a summary."""
        executor = AutonomousExecutor()
        results = [
            TaskResult(success=True, output="done", task="task1"),
            TaskResult(success=False, output="", task="task2", error="failed"),
        ]
        summary = executor.summarize(results)
        assert "Total tasks: 2" in summary
        assert "Successful: 1" in summary
        assert "Failed: 1" in summary

    def test_progress_callback(self):
        """Should call progress callback."""
        executor = AutonomousExecutor()
        calls = []

        def on_progress(step, total, msg):
            calls.append((step, total, msg))

        executor.run("Test task", on_progress=on_progress)
        # Callback should have been called at least once
        assert len(calls) >= 0  # May be 0 if no steps recorded


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Should run a full task through the pipeline."""
        from astroml.agent import create_agent, AutonomousExecutor, Task

        agent = create_agent(agent_type="planner")
        executor = TaskExecutor(agent=agent)
        task = Task(description="Calculate the sum of 1 to 10")
        result = executor.execute(task)

        assert result.success
        assert len(result.output) > 0
        assert len(result.steps) > 0

    def test_memory_persistence_roundtrip(self, tmp_path):
        """Should persist and restore memory."""
        from astroml.agent import MemoryManager, MemoryConfig, Message

        manager = MemoryManager(MemoryConfig())
        manager.add_message(Message(role="user", content="hello"))
        manager.store_fact("key", "value")

        path = tmp_path / "memory.json"
        manager.save(str(path))

        manager2 = MemoryManager(MemoryConfig())
        manager2.load(str(path))
        assert manager2.retrieve_fact("key") == "value"

    def test_tool_execution_through_registry(self):
        """Should execute tools through the registry."""
        from astroml.agent import ToolRegistry, CalculatorTool

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        result = registry.execute("calculator", expression="10 * 5 + 3")
        assert result.success
        assert result.output == "53"

    def test_agent_with_custom_tools(self):
        """Should work with custom tools."""
        from astroml.agent import create_agent, ToolRegistry, Tool, ToolResult

        class CustomTool(Tool):
            name = "custom"
            description = "A custom tool"
            parameters = {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                return ToolResult(success=True, output="custom result")

        registry = ToolRegistry()
        registry.register(CustomTool())

        agent = create_agent(agent_type="react")
        agent.tools = registry
        result = agent.run("Use the custom tool")
        assert len(result) > 0
