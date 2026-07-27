"""Configuration dataclasses for the LLM Agent Framework.

All configuration objects are immutable (``frozen=True``) dataclasses so
they can be safely shared across threads and cached.  Settings can be
loaded from environment variables, YAML files, or constructed directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM provider.

    Attributes:
        provider: Provider type — ``"openai"``, ``"anthropic"``, or
            ``"mock"`` (the default, which uses a deterministic stub
            suitable for testing without API access).
        model: Model name / identifier.
        api_key: API key (read from env if not provided).
        api_base: Optional custom API base URL.
        max_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        timeout_seconds: HTTP request timeout.
        max_retries: Number of retry attempts on transient failures.
    """

    provider: str = "mock"
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 3

    @classmethod
    def from_env(cls, prefix: str = "ASTROML_AGENT") -> "LLMConfig":
        """Build an :class:`LLMConfig` from environment variables.

        Recognised variables (all optional):
            ``{prefix}_LLM_PROVIDER``, ``{prefix}_LLM_MODEL``,
            ``{prefix}_LLM_API_KEY``, ``{prefix}_LLM_API_BASE``,
            ``{prefix}_LLM_MAX_TOKENS``, ``{prefix}_LLM_TEMPERATURE``,
            ``{prefix}_LLM_TOP_P``, ``{prefix}_LLM_TIMEOUT``.
        """
        def _get(key: str, default: Any = None) -> Any:
            return os.environ.get(f"{prefix}_{key}", default)

        return cls(
            provider=_get("LLM_PROVIDER", "mock"),
            model=_get("LLM_MODEL", DEFAULT_MODEL),
            api_key=_get("LLM_API_KEY"),
            api_base=_get("LLM_API_BASE"),
            max_tokens=int(_get("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            temperature=float(_get("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
            top_p=float(_get("LLM_TOP_P", DEFAULT_TOP_P)),
            timeout_seconds=float(_get("LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)),
            max_retries=int(_get("LLM_MAX_RETRIES", 3)),
        )


# ---------------------------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------------------------

DEFAULT_SHORT_TERM_LIMIT = 20
DEFAULT_LONG_TERM_LIMIT = 1000
DEFAULT_EPISODE_LIMIT = 100


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for agent memory subsystems.

    Attributes:
        short_term_limit: Maximum number of messages retained in short-term
            (sliding-window) memory.
        long_term_limit: Maximum number of entries in long-term memory.
        episode_limit: Maximum number of completed episodes to retain.
        persist_long_term: Whether to persist long-term memory to disk.
        persist_path: Path to the persistence file.
    """

    short_term_limit: int = DEFAULT_SHORT_TERM_LIMIT
    long_term_limit: int = DEFAULT_LONG_TERM_LIMIT
    episode_limit: int = DEFAULT_EPISODE_LIMIT
    persist_long_term: bool = False
    persist_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Executor configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_STEP_TIMEOUT = 120.0
DEFAULT_TASK_TIMEOUT = 600.0


@dataclass(frozen=True)
class ExecutorConfig:
    """Configuration for the autonomous task executor.

    Attributes:
        max_steps: Maximum reasoning steps per task before raising
            :class:`~astroml.agent.exceptions.MaxStepsExceededError`.
        max_iterations: Maximum number of plan-execute-refine iterations.
        step_timeout: Per-step timeout in seconds.
        task_timeout: Overall task timeout in seconds.
        auto_approve_tools: If ``True``, tools are executed without
            human confirmation.
        recovery_attempts: Number of times to retry a failed step
            with a different approach.
    """

    max_steps: int = DEFAULT_MAX_STEPS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    step_timeout: float = DEFAULT_STEP_TIMEOUT
    task_timeout: float = DEFAULT_TASK_TIMEOUT
    auto_approve_tools: bool = True
    recovery_attempts: int = 2


# ---------------------------------------------------------------------------
# Top-level agent configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Top-level configuration for an :class:`~astroml.agent.base.Agent`.

    Aggregates :class:`LLMConfig`, :class:`MemoryConfig`, and
    :class:`ExecutorConfig` into a single immutable object.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    agent_type: str = "react"  # "react", "cot", "planner"
    name: str = "astroml-agent"
    description: str = "Autonomous LLM agent for AstroML tasks"
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Build an :class:`AgentConfig` from a nested dictionary.

        Unknown keys are silently ignored so partial configs work.
        """
        def _build(sub: Dict[str, Any], sub_cls: type) -> Any:
            valid = {f.name for f in fields(sub_cls)}
            filtered = {k: v for k, v in sub.items() if k in valid}
            return sub_cls(**filtered)

        llm_data = data.get("llm", {})
        mem_data = data.get("memory", {})
        exec_data = data.get("executor", {})

        return cls(
            llm=_build(llm_data, LLMConfig),
            memory=_build(mem_data, MemoryConfig),
            executor=_build(exec_data, ExecutorConfig),
            agent_type=data.get("agent_type", "react"),
            name=data.get("name", "astroml-agent"),
            description=data.get("description", "Autonomous LLM agent for AstroML tasks"),
            verbose=data.get("verbose", False),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """Load configuration from a YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = "ASTROML_AGENT") -> "AgentConfig":
        """Build a complete :class:`AgentConfig` from environment variables."""
        return cls(
            llm=LLMConfig.from_env(prefix),
            memory=MemoryConfig(),
            executor=ExecutorConfig(),
            agent_type=os.environ.get(f"{prefix}_TYPE", "react"),
            name=os.environ.get(f"{prefix}_NAME", "astroml-agent"),
            description=os.environ.get(f"{prefix}_DESCRIPTION", "Autonomous LLM agent for AstroML tasks"),
            verbose=os.environ.get(f"{prefix}_VERBOSE", "").lower() in ("1", "true", "yes"),
        )
