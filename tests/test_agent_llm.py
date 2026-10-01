"""Tests for the provider agnostic LLM layer."""
from __future__ import annotations

import asyncio

import pytest

from astroml.agent.llm import (
    CallableLLM,
    EchoLLM,
    LLMProvider,
    LLMResponse,
    OpenAICompatibleLLM,
    ScriptedLLM,
    ScriptExhaustedError,
    provider_from_env,
)
from astroml.agent.types import Message, ToolCall, ToolSpec


def _run(coro):
    """Execute a coroutine from a synchronous test."""
    return asyncio.run(coro)


class TestScriptedLLM:
    def test_returns_responses_in_order(self):
        llm = ScriptedLLM(["first", "second"])
        assert _run(llm.complete([Message.user("hi")])).content == "first"
        assert _run(llm.complete([Message.user("hi")])).content == "second"

    def test_records_every_prompt(self):
        llm = ScriptedLLM(["ok"])
        messages = [Message.system("s"), Message.user("u")]
        _run(llm.complete(messages))
        assert llm.calls[0] == messages

    def test_coerces_dict_responses_with_tool_calls(self):
        llm = ScriptedLLM(
            [
                {
                    "content": "thinking",
                    "tool_calls": [
                        {"id": "1", "name": "t", "arguments": {"a": 1}}
                    ],
                }
            ]
        )
        response = _run(llm.complete([Message.user("hi")]))
        assert response.content == "thinking"
        assert response.tool_calls == [ToolCall(id="1", name="t", arguments={"a": 1})]
        assert response.has_tool_calls is True

    def test_coerces_callable_responses(self):
        def responder(messages, tools):
            return LLMResponse(content=f"seen {len(tools)} tools")

        llm = ScriptedLLM([responder])
        response = _run(
            llm.complete(
                [Message.user("hi")],
                tools=[ToolSpec(name="a", description="d")],
            )
        )
        assert response.content == "seen 1 tools"

    def test_token_usage_is_passed_through(self):
        llm = ScriptedLLM(
            [{"content": "x", "prompt_tokens": 3, "completion_tokens": 4}]
        )
        response = _run(llm.complete([Message.user("hi")]))
        assert response.total_tokens == 7

    def test_exhaustion_raises_by_default(self):
        llm = ScriptedLLM(["only"])
        _run(llm.complete([Message.user("hi")]))
        with pytest.raises(ScriptExhaustedError):
            _run(llm.complete([Message.user("hi")]))

    def test_repeat_last_replays_final_response(self):
        llm = ScriptedLLM(["only"], on_exhausted="repeat_last")
        _run(llm.complete([Message.user("hi")]))
        assert _run(llm.complete([Message.user("hi")])).content == "only"

    def test_empty_exhaustion_returns_blank_response(self):
        llm = ScriptedLLM([], on_exhausted="empty")
        assert _run(llm.complete([Message.user("hi")])).content == ""

    def test_rejects_invalid_on_exhausted(self):
        with pytest.raises(ValueError):
            ScriptedLLM([], on_exhausted="bogus")

    def test_rejects_unsupported_response_types(self):
        llm = ScriptedLLM([42])
        with pytest.raises(TypeError):
            _run(llm.complete([Message.user("hi")]))


class TestEchoLLM:
    def test_echoes_last_user_message(self):
        response = _run(
            EchoLLM().complete(
                [Message.system("s"), Message.user("first"), Message.user("second")]
            )
        )
        assert response.content == "Echo: second"

    def test_falls_back_to_last_non_system_message(self):
        response = _run(
            EchoLLM().complete([Message.system("s"), Message.assistant("tool output")])
        )
        assert response.content == "Echo: tool output"

    def test_acomplete_alias_works(self):
        assert _run(EchoLLM().acomplete([Message.user("hi")])).content == "Echo: hi"


class TestCallableLLM:
    def test_supports_sync_callables(self):
        llm = CallableLLM(lambda messages, tools, temperature, max_tokens: "sync ok")
        assert _run(llm.complete([Message.user("hi")])).content == "sync ok"

    def test_supports_async_callables(self):
        async def responder(messages, tools, temperature, max_tokens):
            return {"content": "async ok"}

        assert _run(CallableLLM(responder).complete([Message.user("hi")])).content == "async ok"

    def test_forwards_sampling_arguments(self):
        seen = {}

        def responder(messages, tools, temperature, max_tokens):
            seen["temperature"] = temperature
            seen["max_tokens"] = max_tokens
            return "ok"

        _run(CallableLLM(responder).complete([Message.user("hi")], temperature=0.7, max_tokens=9))
        assert seen == {"temperature": 0.7, "max_tokens": 9}


class TestLLMProviderInterface:
    def test_abstract_provider_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMProvider()


class TestOpenAICompatibleLLM:
    def _llm(self, **kwargs) -> OpenAICompatibleLLM:
        return OpenAICompatibleLLM(
            "test-model", base_url="http://localhost:9999/v1/", **kwargs
        )

    def test_endpoint_strips_trailing_slash(self):
        assert self._llm().endpoint == "http://localhost:9999/v1/chat/completions"

    def test_rejects_empty_model(self):
        with pytest.raises(ValueError):
            OpenAICompatibleLLM("")

    def test_build_payload_includes_tools_and_sampling(self):
        payload = self._llm().build_payload(
            [Message.system("s"), Message.user("u")],
            tools=[
                ToolSpec(
                    name="count", description="d", parameters={"type": "object"}
                )
            ],
            temperature=0.2,
            max_tokens=64,
        )
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 64
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "count"
        assert payload["messages"][0] == {"role": "system", "content": "s"}

    def test_build_payload_omits_optional_sections(self):
        payload = self._llm().build_payload([Message.user("u")])
        assert "tools" not in payload
        assert "max_tokens" not in payload

    def test_parse_response_reads_message_and_usage(self):
        response = self._llm().parse_response(
            {
                "model": "served-model",
                "choices": [
                    {
                        "message": {
                            "content": "hello",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "function": {
                                        "name": "count",
                                        "arguments": '{"a": 1}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            }
        )
        assert response.content == "hello"
        assert response.model == "served-model"
        assert response.tool_calls[0].name == "count"
        assert response.tool_calls[0].arguments == {"a": 1}
        assert response.total_tokens == 16

    def test_parse_response_handles_missing_choices(self):
        response = self._llm().parse_response({"choices": []})
        assert response.content == ""
        assert response.model == "test-model"


class TestProviderFromEnv:
    def test_defaults_to_echo(self):
        assert isinstance(provider_from_env({}), EchoLLM)

    def test_ollama_defaults(self):
        provider = provider_from_env({"ASTROML_LLM_PROVIDER": "ollama"})
        assert isinstance(provider, OpenAICompatibleLLM)
        assert provider.base_url == "http://localhost:11434/v1"
        assert provider.model == "llama3.1"

    def test_openai_defaults_and_overrides(self):
        provider = provider_from_env(
            {
                "ASTROML_LLM_PROVIDER": "openai",
                "ASTROML_LLM_MODEL": "gpt-4o",
                "ASTROML_LLM_BASE_URL": "https://example.test/v1",
                "ASTROML_LLM_API_KEY": "secret",
            }
        )
        assert isinstance(provider, OpenAICompatibleLLM)
        assert provider.model == "gpt-4o"
        assert provider.endpoint == "https://example.test/v1/chat/completions"
        assert provider.api_key == "secret"

    def test_scripted_provider_uses_script(self):
        provider = provider_from_env(
            {
                "ASTROML_LLM_PROVIDER": "scripted",
                "ASTROML_LLM_SCRIPT": "first||second",
                "ASTROML_LLM_ON_EXHAUSTED": "raise",
            }
        )
        assert _run(provider.complete([Message.user("hi")])).content == "first"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown ASTROML_LLM_PROVIDER"):
            provider_from_env({"ASTROML_LLM_PROVIDER": "bogus"})

    def test_invalid_timeout_falls_back_to_default(self):
        provider = provider_from_env(
            {"ASTROML_LLM_PROVIDER": "openai", "ASTROML_LLM_TIMEOUT": "nope"}
        )
        assert provider.timeout == 60.0
