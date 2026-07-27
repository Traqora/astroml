"""Custom exceptions for the LLM Agent Framework.

All agent-related errors derive from :class:`AgentError` so callers can
catch every framework-specific exception with a single ``except`` clause
while still distinguishing individual failure modes when needed.
"""
from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent framework errors."""


class LLMError(AgentError):
    """Raised when an LLM provider call fails or returns an invalid response."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM provider is misconfigured (e.g. missing API key)."""


class ToolError(AgentError):
    """Raised when a tool fails to execute or returns an error result."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered in the tool registry."""


class MemoryError(AgentError):
    """Raised when a memory operation fails (e.g. capacity exceeded).

    Note: this intentionally shadows the built-in ``MemoryError`` only within
    the agent namespace; the built-in is still accessible via ``builtins``.
    """


class PlanningError(AgentError):
    """Raised when task planning or decomposition fails."""


class MaxStepsExceededError(AgentError):
    """Raised when the agent exceeds the configured maximum number of reasoning steps."""


class TaskTimeoutError(AgentError):
    """Raised when an autonomous task exceeds its time budget."""


class TaskFailedError(AgentError):
    """Raised when a task cannot be completed and no further recovery is possible."""
