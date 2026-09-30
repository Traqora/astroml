"""LLM agent framework for AstroML.

Multi-step reasoning and autonomous task execution on top of a provider
agnostic LLM interface, a typed tool registry and bounded conversation
memory.  Quick start::

    from astroml.agent import AgentConfig, AgentExecutor, ToolRegistry, tool

    @tool()
    def edge_count(edges: list) -> int:
        \"\"\"Count the transactions in the supplied graph.\"\"\"
        return len(edges)

    agent = AgentExecutor(
        llm=my_provider,                     # see astroml.agent.llm
        tools=ToolRegistry([edge_count]),
        config=AgentConfig(max_steps=4, mode="react"),
    )
    result = agent.run("How many transactions are there?")

``mode="react"`` drives models without native tool calling; ``mode="native"``
uses OpenAI style function calling; ``mode="auto"`` supports both.  See
``docs/agent-framework.md`` for the full guide.
"""
from .compression import (
    CompressedPrompt,
    CompressionConfig,
    CompressionStats,
    Compressor,
    DEFAULT_STRATEGIES,
    MessageGroup,
    ObservationDedupe,
    PromptCompressor,
    ToolOutputCompressor,
    WhitespaceNormalizer,
    compress_text,
    compress_text_batch,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_messages_tokens_parallel,
    estimate_tokens,
    extractive_summary,
    group_messages,
    normalize_whitespace,
    total_stats,
    truncate_text,
)
from .executor import DEFAULT_SYSTEM_PROMPT, AgentExecutor, format_tool_catalogue
from .llm import (
    CallableLLM,
    EchoLLM,
    LLMProvider,
    LLMResponse,
    OpenAICompatibleLLM,
    ScriptExhaustedError,
    ScriptedLLM,
    provider_from_env,
)
from .memory import ConversationMemory, Memory
from .planner import (
    Plan,
    PlanStep,
    ReActOutput,
    ReActParser,
    TaskPlanner,
    extract_json_block,
    extract_json_blocks_parallel,
)
from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    ComparisonResult,
    benchmark,
    benchmark_parallel_vs_sequential,
    compare_benchmarks,
    timer,
)
from .tools import (
    Tool,
    ToolError,
    ToolRegistry,
    render_result,
    result_to_message,
    tool,
    tool_from_callable,
)
from .types import (
    VALID_MODES,
    AgentConfig,
    AgentRunResult,
    AgentStep,
    AgentTrace,
    Message,
    Role,
    StepStatus,
    ToolCall,
    ToolResult,
    ToolSpec,
)

__all__ = [
    # core loop
    "AgentConfig",
    "AgentExecutor",
    "AgentRunResult",
    "AgentStep",
    "AgentTrace",
    "DEFAULT_SYSTEM_PROMPT",
    "StepStatus",
    "VALID_MODES",
    "format_tool_catalogue",
    # types
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    # providers
    "CallableLLM",
    "EchoLLM",
    "LLMProvider",
    "LLMResponse",
    "OpenAICompatibleLLM",
    "ScriptExhaustedError",
    "ScriptedLLM",
    "provider_from_env",
    # tools
    "Tool",
    "ToolError",
    "ToolRegistry",
    "render_result",
    "result_to_message",
    "tool",
    "tool_from_callable",
    # benchmarking
    "BenchmarkResult",
    "BenchmarkSuite",
    "ComparisonResult",
    "benchmark",
    "benchmark_parallel_vs_sequential",
    "compare_benchmarks",
    "timer",
    # memory and planning
    "ConversationMemory",
    "Memory",
    "Plan",
    "PlanStep",
    "ReActOutput",
    "ReActParser",
    "TaskPlanner",
    "extract_json_block",
    "extract_json_blocks_parallel",
    # prompt compression
    "CompressedPrompt",
    "CompressionConfig",
    "CompressionStats",
    "Compressor",
    "DEFAULT_STRATEGIES",
    "MessageGroup",
    "ObservationDedupe",
    "PromptCompressor",
    "ToolOutputCompressor",
    "WhitespaceNormalizer",
    "compress_text",
    "compress_text_batch",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_messages_tokens_parallel",
    "estimate_tokens",
    "extractive_summary",
    "group_messages",
    "normalize_whitespace",
    "total_stats",
    "truncate_text",
    # domain tools
    "DOMAIN_TOOLS",
    "anomaly_score_tool",
    "build_default_registry",
]


#: Domain tool names re-exported lazily by :func:`__getattr__`.
_DOMAIN_EXPORTS = frozenset({"DOMAIN_TOOLS", "anomaly_score_tool", "build_default_registry"})


def __getattr__(name: str):
    """Lazily re-export the AstroML domain tools.

    ``astroml.agent.domain_tools`` depends on ``astroml.features`` (which
    eagerly imports the SQLAlchemy backed pipelines). Resolving it on first
    access keeps ``import astroml.agent`` lightweight so the agent loop can
    run in minimal environments.
    """
    if name in _DOMAIN_EXPORTS:
        from . import domain_tools

        return getattr(domain_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
