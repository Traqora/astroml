"""LLM Agent Framework for AstroML.

Provides multi-step reasoning and autonomous task execution capabilities
through a modular agent architecture.

Core components:
    - :class:`~astroml.agent.base.Agent` — abstract base agent
    - :class:`~astroml.agent.base.ReActAgent` — ReAct reasoning agent
    - :class:`~astroml.agent.base.ChainOfThoughtAgent` — chain-of-thought agent
    - :class:`~astroml.agent.base.PlannerAgent` — planning agent
    - :class:`~astroml.agent.executor.TaskExecutor` — task execution engine
    - :class:`~astroml.agent.executor.AutonomousExecutor` — autonomous executor
    - :class:`~astroml.agent.memory.MemoryManager` — memory management
    - :class:`~astroml.agent.tools.ToolRegistry` — tool registry
    - :class:`~astroml.agent.llm.LLMClient` — LLM provider abstraction

Quick start::

    from astroml.agent import create_agent, AutonomousExecutor

    executor = AutonomousExecutor()
    result = executor.run("Calculate the area of a circle with radius 5")
    print(result.output)
"""
from __future__ import annotations

from .base import (
    Agent,
    ChainOfThoughtAgent,
    ExecutionPlan,
    PlanStep,
    PlannerAgent,
    ReActAgent,
    StepResult,
    create_agent,
)
from .config import (
    AgentConfig,
    ExecutorConfig,
    LLMConfig,
    MemoryConfig,
)
from .exceptions import (
    AgentError,
    LLMConfigurationError,
    LLMError,
    MaxStepsExceededError,
    MemoryError as AgentMemoryError,
    PlanningError,
    TaskFailedError,
    TaskTimeoutError,
    ToolError,
    ToolNotFoundError,
)
from .executor import (
    AutonomousExecutor,
    Task,
    TaskExecutor,
    TaskResult,
)
from .llm import (
    LLMClient,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
)
from .memory import (
    Episode,
    EpisodicMemory,
    LongTermMemory,
    MemoryEntry,
    MemoryManager,
    Message,
    ShortTermMemory,
)
from .tools import (
    CalculatorTool,
    FileReadTool,
    FileWriteTool,
    HTTPRequestTool,
    ListDirectoryTool,
    PythonREPLTool,
    SearchTool,
    Tool,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)

__all__ = [
    # Base agent
    "Agent",
    "ReActAgent",
    "ChainOfThoughtAgent",
    "PlannerAgent",
    "StepResult",
    "ExecutionPlan",
    "PlanStep",
    "create_agent",
    # Configuration
    "AgentConfig",
    "LLMConfig",
    "MemoryConfig",
    "ExecutorConfig",
    # Exceptions
    "AgentError",
    "LLMError",
    "LLMConfigurationError",
    "ToolError",
    "ToolNotFoundError",
    "AgentMemoryError",
    "PlanningError",
    "MaxStepsExceededError",
    "TaskTimeoutError",
    "TaskFailedError",
    # Executor
    "TaskExecutor",
    "AutonomousExecutor",
    "Task",
    "TaskResult",
    # LLM
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    # Memory
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
    "EpisodicMemory",
    "Message",
    "MemoryEntry",
    "Episode",
    # Tools
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "CalculatorTool",
    "PythonREPLTool",
    "FileReadTool",
    "FileWriteTool",
    "ListDirectoryTool",
    "SearchTool",
    "HTTPRequestTool",
    "create_default_registry",
]

__version__ = "0.1.0"
