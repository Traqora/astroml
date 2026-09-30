"""Provider agnostic LLM interface for the AstroML agent framework.

An agent only requires an object implementing :class:`LLMProvider`. Three
ready made providers ship with the framework so that the loop can be used
offline, against a custom backend, or against any OpenAI compatible HTTP
endpoint:

* :class:`ScriptedLLM` — deterministic replay of a fixed response script
  (used by the test suite and for offline demos).
* :class:`CallableLLM` — adapts any ``messages -> response`` callable.
* :class:`EchoLLM` — trivial offline default that echoes the prompt.
* :class:`OpenAICompatibleLLM` — HTTP provider for OpenAI, Ollama
  (``http://localhost:11434/v1``), vLLM, LM Studio, OpenRouter, ...

Use :func:`provider_from_env` to pick a provider from environment variables
(``ASTROML_LLM_PROVIDER``, ``ASTROML_LLM_MODEL``, ``ASTROML_LLM_BASE_URL``,
``ASTROML_LLM_API_KEY``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .types import Message, Role, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0


@dataclass
class LLMResponse:
    """Normalised completion returned by every provider."""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if self.model:
            payload["model"] = self.model
        return payload


class LLMProvider(ABC):
    """Minimal async interface every LLM backend must implement."""

    #: Short, human readable provider name used in traces and logs.
    name: str = "llm"
    #: Default model identifier, if the provider has one.
    model: Optional[str] = None

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Return the model's next turn given the conversation *messages*.

        Implementations must never raise for a "normal" model refusal — they
        should return an :class:`LLMResponse` whose ``content`` explains the
        situation.  Transport errors may propagate and are handled by the
        executor.
        """

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Alias of :meth:`complete` kept for readability at call sites."""
        return await self.complete(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )


class ScriptExhaustedError(RuntimeError):
    """Raised when a :class:`ScriptedLLM` runs out of scripted responses."""


class ScriptedLLM(LLMProvider):
    """Deterministic provider replaying a fixed script of responses.

    Responses may be :class:`LLMResponse` objects, plain strings, dicts, or
    callables ``(messages, tools) -> LLMResponse | str`` (sync or async).
    Callables are useful for scripted runs that must react to the prompt.

    Args:
        responses: Ordered responses returned by successive calls.
        on_exhausted: What to do once the script is consumed —
            ``"raise"`` (default) raises :class:`ScriptExhaustedError`,
            ``"repeat_last"`` replays the final response forever,
            ``"empty"`` returns an empty :class:`LLMResponse`.
        name: Provider name reported in traces.
    """

    def __init__(
        self,
        responses: Sequence[Any],
        *,
        on_exhausted: str = "raise",
        name: str = "scripted",
        model: Optional[str] = None,
    ) -> None:
        if on_exhausted not in ("raise", "repeat_last", "empty"):
            raise ValueError(
                "on_exhausted must be 'raise', 'repeat_last' or 'empty'"
            )
        self.responses: List[Any] = list(responses)
        self.on_exhausted = on_exhausted
        self.name = name
        self.model = model
        #: Every prompt the provider has been called with (useful in tests).
        self.calls: List[List[Message]] = []
        self._cursor = 0

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))

        if not self.responses:
            if self.on_exhausted == "empty":
                return LLMResponse(model=self.model)
            raise ScriptExhaustedError("ScriptedLLM received no responses")

        index = self._cursor
        self._cursor += 1
        if index >= len(self.responses):
            if self.on_exhausted == "raise":
                raise ScriptExhaustedError(
                    f"ScriptedLLM exhausted after {len(self.responses)} responses"
                )
            if self.on_exhausted == "empty":
                return LLMResponse(model=self.model)
            index = len(self.responses) - 1

        item = self.responses[index]
        if callable(item):
            item = item(list(messages), list(tools or []))
            if asyncio.iscoroutine(item):
                item = await item
        return _coerce_response(item, model=self.model)


def _coerce_response(item: Any, *, model: Optional[str] = None) -> LLMResponse:
    """Normalise scripted/custom responses into an :class:`LLMResponse`."""
    if isinstance(item, LLMResponse):
        return item
    if isinstance(item, str):
        return LLMResponse(content=item, model=model)
    if isinstance(item, Mapping):
        raw = dict(item)
        tool_calls = [
            call if isinstance(call, ToolCall) else ToolCall.from_dict(call)
            for call in raw.pop("tool_calls", []) or []
        ]
        return LLMResponse(
            content=str(raw.pop("content", "") or ""),
            tool_calls=tool_calls,
            model=raw.pop("model", model),
            prompt_tokens=int(raw.pop("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.pop("completion_tokens", 0) or 0),
            raw=raw or None,
        )
    raise TypeError(
        "Scripted/callable responses must be LLMResponse, str, dict or callable; "
        f"got {type(item).__name__}"
    )


class EchoLLM(LLMProvider):
    """Offline default provider that echoes the latest user message.

    Useful as a dependency free placeholder: the agent loop, tool registry
    and tracing can all be exercised without a real model.
    """

    def __init__(self, *, prefix: str = "Echo: ", name: str = "echo") -> None:
        self.prefix = prefix
        self.name = name
        self.model = None

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        last_user = ""
        for message in messages:
            if message.role is Role.USER:
                last_user = message.content
        if not last_user:
            for message in reversed(list(messages)):
                if message.role is not Role.SYSTEM:
                    last_user = message.content
                    break
        return LLMResponse(content=f"{self.prefix}{last_user}")


class CallableLLM(LLMProvider):
    """Adapt any ``messages -> LLMResponse`` function into a provider.

    The callable receives keyword arguments ``messages``, ``tools``,
    ``temperature`` and ``max_tokens`` and may be synchronous or async.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str = "callable",
        model: Optional[str] = None,
    ) -> None:
        self._func = func
        self.name = name
        self.model = model

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        result = self._func(
            messages=list(messages),
            tools=list(tools or []),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if asyncio.iscoroutine(result):
            result = await result
        return _coerce_response(result, model=self.model)


class OpenAICompatibleLLM(LLMProvider):
    """Chat-completions provider for any OpenAI compatible HTTP endpoint.

    Tested against the ``/v1/chat/completions`` contract shared by OpenAI,
    Ollama (``http://localhost:11434/v1``), vLLM, LM Studio and OpenRouter.

    ``aiohttp`` (already a project dependency) is imported lazily so the rest
    of the agent framework keeps working when it is not installed.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Optional[Mapping[str, str]] = None,
        name: str = "openai-compatible",
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.extra_headers = dict(headers or {})
        self.name = name

    @property
    def endpoint(self) -> str:
        """Full chat completions URL."""
        return f"{self.base_url}/chat/completions"

    def build_payload(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the JSON request body (exposed for testing)."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai_dict() for message in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if tools:
            payload["tools"] = [spec.to_openai_tool() for spec in tools]
            payload["tool_choice"] = "auto"
        return payload

    def parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        """Convert a chat-completions body into an :class:`LLMResponse`."""
        choices = list(data.get("choices") or [])
        model = data.get("model") or self.model
        if not choices:
            return LLMResponse(content="", model=model, raw=data)

        message = dict(choices[0].get("message") or {})
        usage = dict(data.get("usage") or {})
        tool_calls = [
            call if isinstance(call, ToolCall) else ToolCall.from_dict(call)
            for call in message.get("tool_calls") or []
        ]
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=data,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = self.build_payload(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        data = await self._post(payload)
        return self.parse_response(data)

    async def _post(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "aiohttp is required for OpenAICompatibleLLM "
                "(install it with `pip install aiohttp`)"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.endpoint, json=dict(payload), headers=headers
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"{self.name} request failed with HTTP {response.status}: "
                        f"{text[:500]}"
                    )
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{self.name} returned a non-JSON body: {text[:500]}"
                    ) from exc


def provider_from_env(env: Optional[Mapping[str, str]] = None) -> LLMProvider:
    """Build a provider from environment variables.

    Recognised variables:

    ``ASTROML_LLM_PROVIDER``  ``echo`` (default), ``scripted``, ``openai``,
                              ``openai-compatible``, ``ollama``, ``vllm``,
                              ``lmstudio``
    ``ASTROML_LLM_MODEL``     model identifier
    ``ASTROML_LLM_BASE_URL``  API root, e.g. ``http://localhost:11434/v1``
    ``ASTROML_LLM_API_KEY``   bearer token, when the endpoint needs one
    ``ASTROML_LLM_TIMEOUT``   request timeout in seconds
    ``ASTROML_LLM_SCRIPT``    ``scripted`` provider script, ``||`` separated
    """
    values: Mapping[str, str] = dict(os.environ if env is None else env)
    kind = (values.get("ASTROML_LLM_PROVIDER") or "").strip().lower()
    model = (values.get("ASTROML_LLM_MODEL") or "").strip() or None
    base_url = (values.get("ASTROML_LLM_BASE_URL") or "").strip() or None
    api_key = (values.get("ASTROML_LLM_API_KEY") or "").strip() or None

    if kind in ("", "echo"):
        return EchoLLM()

    if kind == "scripted":
        script = values.get("ASTROML_LLM_SCRIPT") or ""
        responses = [chunk for chunk in script.split("||") if chunk.strip()]
        return ScriptedLLM(
            responses or [""],
            on_exhausted=values.get("ASTROML_LLM_ON_EXHAUSTED", "repeat_last"),
        )

    if kind in (
        "openai",
        "openai-compatible",
        "openai_compatible",
        "ollama",
        "vllm",
        "lmstudio",
    ):
        if kind == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            model = model or "llama3.1"
        elif kind in ("vllm", "lmstudio"):
            base_url = base_url or "http://localhost:8000/v1"
            model = model or "default"
        else:
            base_url = base_url or "https://api.openai.com/v1"
            model = model or "gpt-4o-mini"
        try:
            timeout = float(values.get("ASTROML_LLM_TIMEOUT", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT
        return OpenAICompatibleLLM(
            model, base_url=base_url, api_key=api_key, timeout=timeout
        )

    raise ValueError(f"Unknown ASTROML_LLM_PROVIDER: {kind!r}")

