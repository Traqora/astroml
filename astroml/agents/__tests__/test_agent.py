"""Tests for the Agent framework."""

import pytest

from astroml.agents.agent import Agent, AgentConfig, AgentStep, AgentResult
from astroml.agents.memory import ConversationMemory, MemoryStore
from astroml.agents.planning import HeuristicPlanner, Task, TaskPlan, TaskStatus
from astroml.agents.tools import DataLoader, ToolRegistry


def test_agent_config():
    """Test AgentConfig creation."""
    config = AgentConfig(
        name="test-agent",
        model="gpt-4",
        temperature=0.5,
        max_steps=5,
    )
    assert config.name == "test-agent"
    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    assert config.max_steps == 5


def test_agent_step():
    """Test AgentStep creation."""
    step = AgentStep(
        step_number=1,
        thought="I need to load the data",
        action="load_data",
        tool_call={"tool": "load_data", "input": {"source": "data.csv"}},
        observation="Data loaded successfully",
    )
    assert step.step_number == 1
    assert step.thought == "I need to load the data"
    assert step.action == "load_data"
    assert step.tool_call == {"tool": "load_data", "input": {"source": "data.csv"}}
    assert step.observation == "Data loaded successfully"


def test_memory_store():
    """Test MemoryStore basic operations."""
    memory = MemoryStore()

    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi there")

    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there"

    recent = memory.get_recent_messages(n=1)
    assert len(recent) == 1
    assert recent[0].content == "Hi there"

    memory.clear()
    assert len(memory.get_messages()) == 0


def test_conversation_memory():
    """Test ConversationMemory with role-based operations."""
    memory = ConversationMemory(max_history=10)

    memory.add_message("user", "Task 1")
    memory.add_message("assistant", "Response 1")
    memory.add_message("user", "Task 2")
    memory.add_message("assistant", "Response 2")

    user_msgs = memory.get_user_messages()
    assert len(user_msgs) == 2

    assistant_msgs = memory.get_assistant_messages()
    assert len(assistant_msgs) == 2

    pairs = memory.get_dialogue_pairs()
    assert len(pairs) == 2
    assert pairs[0][0].content == "Task 1"
    assert pairs[0][1].content == "Response 1"


def test_tool_registry():
    """Test ToolRegistry operations."""
    registry = ToolRegistry()

    tool = DataLoader()
    registry.register(tool)

    assert "load_data" in registry.list_tools()
    assert registry.get_tool("load_data") is not None
    assert registry.get_tool("nonexistent") is None

    registry.unregister("load_data")
    assert "load_data" not in registry.list_tools()


def test_heuristic_planner():
    """Test HeuristicPlanner task decomposition."""
    planner = HeuristicPlanner()

    plan = planner.plan("Analyze the transaction data for fraud patterns")
    assert plan.description == "Analyze the transaction data for fraud patterns"
    assert len(plan.tasks) > 1
    assert plan.tasks[0].status == TaskStatus.PENDING

    plan2 = planner.plan("Train a fraud detection model")
    assert len(plan2.tasks) > 1
    assert any("train" in task.description.lower() for task in plan2.tasks)


def test_agent_without_llm():
    """Test Agent execution without LLM (mock mode)."""
    config = AgentConfig(
        name="test-agent",
        max_steps=3,
        enable_planning=False,
        enable_memory=False,
        verbose=False,
    )

    tools = [DataLoader()]
    agent = Agent(config=config, tools=tools)

    result = agent.run("Load data from data.csv")

    assert isinstance(result, AgentResult)
    assert len(result.steps) > 0
    assert result.steps[0].step_number == 1


def test_task_plan():
    """Test TaskPlan operations."""
    plan = TaskPlan(
        description="Test plan",
        tasks=[
            Task(id="task1", description="First task"),
            Task(id="task2", description="Second task", dependencies=["task1"]),
        ],
    )

    assert plan.get_progress() == 0.0
    assert not plan.is_complete()

    plan.tasks[0].mark_completed()
    assert plan.get_progress() == 50.0

    plan.tasks[1].mark_completed()
    assert plan.is_complete()
    assert plan.get_progress() == 100.0


def test_task_ready_check():
    """Test Task dependency checking."""
    task1 = Task(id="task1", description="First task")
    task2 = Task(id="task2", description="Second task", dependencies=["task1"])
    task3 = Task(id="task3", description="Third task", dependencies=["task1", "task2"])

    assert task1.is_ready(set())
    assert not task2.is_ready(set())
    assert task2.is_ready({"task1"})
    assert not task3.is_ready({"task1"})
    assert task3.is_ready({"task1", "task2"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
