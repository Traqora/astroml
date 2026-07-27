"""LLM provider abstraction layer.

Provides a uniform interface for chat-completion LLM providers so the
agent framework can switch between OpenAI, Anthropic, and a deterministic
mock without changing any agent code.

Usage::

    from astroml.agent.llm import LLMClient, LLMConfig

    config = LLMConfig(provider="mock")
    client = LLMClient(config)
    response = client.chat([
        {"role": "user", "content": "What is 2 + 2?"}],
    )
    print(response.content)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .config import LLMConfig
from .exceptions import LLMConfigurationError, LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMMessage:
    """A single message in an LLM conversation.

    Attributes:
        role: ``"system"``, ``"user"``, ``"assistant"``, or ``"tool"``.
        content: The message text.
        tool_call_id: For tool-result messages, the ID of the tool call.
        tool_calls: For assistant messages, a list of tool-call dicts.
    """

    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for API payloads."""
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            msg["tool_calls"] = self.tool_calls
        return msg


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The generated text (may be empty if only tool calls).
        model: The model that generated the response.
        usage: Token usage dict (``{"prompt_tokens", "completion_tokens",
            "total_tokens"}``).
        tool_calls: Optional list of tool-call dicts.
        finish_reason: ``"stop"``, ``"tool_calls"``, ``"length"``, etc.
    """

    content: str
    model: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class LLMProvider:
    """Abstract base class for LLM providers.

    Subclasses must implement :meth:`chat`.  The base class handles
    retry logic and error normalisation.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: Sequence[LLMMessage | Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Send *messages* to the LLM and return the response.

        Args:
            messages: Conversation history as :class:`LLMMessage` objects
                or plain dicts.
            tools: Optional list of tool schemas for function calling.

        Returns:
            An :class:`LLMResponse`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _normalise_messages(
        self,
        messages: Sequence[LLMMessage | Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert a mixed sequence of messages to plain dicts."""
        result: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                result.append(msg.to_dict())
            elif isinstance(msg, dict):
                result.append(msg)
            else:
                raise LLMError(f"Unsupported message type: {type(msg).__name__}")
        return result


# ---------------------------------------------------------------------------
# Mock provider — deterministic, no API key required
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Deterministic mock LLM provider for testing and development.

    Returns canned responses based on simple pattern matching so that
    agent logic can be exercised without network access or API keys.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._call_count = 0

    def chat(
        self,
        messages: Sequence[LLMMessage | Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self._call_count += 1
        msgs = self._normalise_messages(messages)

        # Find the last user message
        user_content = ""
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        content = self._generate_response(user_content, tools)

        return LLMResponse(
            content=content,
            model=self.config.model,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            tool_calls=None,
            finish_reason="stop",
        )

    def _generate_response(
        self,
        user_content: str,
        tools: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Generate a deterministic response based on the user query."""
        content_lower = user_content.lower().strip()

        # Planning: detect multi-step task requests
        if any(kw in content_lower for kw in ["plan", "break down", "steps", "task list"]):
            return (
                "I'll break this down into steps:\n"
                "1. Understand the task requirements\n"
                "2. Identify available tools and resources\n"
                "3. Execute each step in sequence\n"
                "4. Verify the results\n"
                "5. Report the final outcome"
            )

        # Calculator-like queries
        if any(kw in content_lower for kw in ["calculate", "compute", "+", "-", "*", "/"]):
            return self._mock_calculate(user_content)

        # Tool-calling: if tools are provided and the query looks actionable
        if tools and any(kw in content_lower for kw in ["read", "write", "list", "run", "execute"]):
            # Return a tool call for the first matching tool
            for tool in tools:
                tool_name = tool.get("function", {}).get("name", "")
                if "read" in tool_name and "read" in content_lower:
                    return self._mock_tool_call(tool_name, {"path": "/tmp/example.txt"})
                if "write" in tool_name and "write" in content_lower:
                    return self._mock_tool_call(tool_name, {"path": "/tmp/output.txt", "content": "result"})
                if "list" in tool_name and "list" in content_lower:
                    return self._mock_tool_call(tool_name, {"path": "."})
                if "python" in tool_name or "exec" in tool_name:
                    return self._mock_tool_call(tool_name, {"code": "print('hello')"})

        # Default: provide a helpful, structured response
        return (
            f"I understand you're asking about: '{user_content}'. "
            "Let me work through this systematically. "
            "I'll use the available tools to investigate and provide a complete answer."
        )

    def _mock_calculate(self, query: str) -> str:
        """Attempt to parse and evaluate a simple arithmetic expression."""
        import re

        # Extract the arithmetic expression
        match = re.search(r"([\d\s\+\-\*\/\(\)\.]+)", query)
        if match:
            expr = match.group(1)
            try:
                # Safe-ish eval for simple arithmetic
                result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
                return f"The result of {expr} is {result}."
            except Exception:
                pass
        return "I can help with calculations. Please provide a clear arithmetic expression."

    def _mock_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Return a string that looks like a tool call (for mock purposes)."""
        return json.dumps({"tool_call": {"name": tool_name, "arguments": args}})


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API.

    Requires the ``openai`` package and a valid API key.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            config = LLMConfig(
                **{**config.__dict__, "api_key": os.environ.get("OPENAI_API_KEY")},
            )
        if not config.api_key:
            raise LLMConfigurationError(
                "OpenAI API key not provided. Set api_key in LLMConfig "
                "or the OPENAI_API_KEY environment variable."
            )
        self._api_key = config.api_key

    def chat(
        self,
        messages: Sequence[LLMMessage | Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        import openai

        msgs = self._normalise_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout_seconds,
                )
                resp = client.chat.completions.create(**kwargs)

                choice = resp.choices[0]
                content = choice.message.content or ""
                tool_calls = None
                if choice.message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                return LLMResponse(
                    content=content,
                    model=resp.model,
                    usage={
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    },
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason or "stop",
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("OpenAI API call failed (attempt %d/%d): %s — retrying in %ds",
                                   attempt + 1, self.config.max_retries, exc, wait)
                    time.sleep(wait)
                else:
                    logger.error("OpenAI API call failed after %d attempts: %s",
                                 self.config.max_retries, exc)

        raise LLMError(f"OpenAI API call failed after {self.config.max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API.

    Requires the ``anthropic`` package and a valid API key.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            config = LLMConfig(
                **{**config.__dict__, "api_key": os.environ.get("ANTHROPIC_API_KEY")},
            )
        if not config.api_key:
            raise LLMConfigurationError(
                "Anthropic API key not provided. Set api_key in LLMConfig "
                "or the ANTHROPIC_API_KEY environment variable."
            )
        self._api_key = config.api_key

    def chat(
        self,
        messages: Sequence[LLMMessage | Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        import anthropic

        msgs = self._normalise_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if tools:
            kwargs["tools"] = tools

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self.config.timeout_seconds,
                )
                resp = client.messages.create(**kwargs)

                content = ""
                tool_calls: Optional[List[Dict[str, Any]]] = None
                for block in resp.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append({
                            "id": block.id,
                            "type": "tool_use",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        })

                return LLMResponse(
                    content=content,
                    model=resp.model,
                    usage={
                        "prompt_tokens": resp.usage.input_tokens,
                        "completion_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                    },
                    tool_calls=tool_calls,
                    finish_reason=resp.stop_reason or "stop",
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Anthropic API call failed (attempt %d/%d): %s — retrying in %ds",
                                   attempt + 1, self.config.max_retries, exc, wait)
                    time.sleep(wait)
                else:
                    logger.error("Anthropic API call failed after %d attempts: %s",
                                 self.config.max_retries, exc)

        raise LLMError(f"Anthropic API call failed after {self.config.max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

# Import os here for the provider classes that reference it
import os  # noqa: E402


_PROVIDER_MAP: Dict[str, type[LLMProvider]] = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


class LLMClient:
    """High-level LLM client that selects and delegates to a provider.

    This is the main entry point for LLM calls within the agent framework.
    It wraps a provider instance and exposes a simple :meth:`chat` method.

    Example::

        client = LLMClient(LLMConfig(provider="mock"))
        resp = client.chat([
            {"role": "user", "content": "Hello!"}],
        )
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        provider_cls = _PROVIDER_MAP.get(config.provider)
        if provider_cls is None:
            raise LLMConfigurationError(
                f"Unknown LLM provider: '{config.provider}'. "
                f"Available: {list(_PROVIDER_MAP.keys())}"
            )
        self.provider: LLMProvider = provider_cls(config)
        self.provider_name = config.provider

    def chat(
        self,
        messages: Sequence[LLMMessage | Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Send a chat request through the configured provider."""
        return self.provider.chat(messages, tools)

    def count_calls(self) -> int:
        """Return the number of calls made (mock provider only)."""
        if hasattr(self.provider, "_call_count"):
            return self.provider._call_count
        return 0
