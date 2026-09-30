"""Tests for the agent framework's core data types."""
from __future__ import annotations

import pytest

from astroml.agent.types import (
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


class TestToolSpec:
    def test_to_dict_preserves_parameters(self):
        spec = ToolSpec(name="count", description="Count things", parameters={"type": "object"})
        payload = spec.to_dict()
        assert payload["name"] == "count"
        assert payload["description"] == "Count things"
        assert payload["parameters"] == {"type": "object"}

    def test_empty_parameters_become_object_schema(self):
        payload = ToolSpec(name="noop", description="Nothing").to_dict()
        assert payload["parameters"] == {"type": "object", "properties": {}}

    def test_to_openai_tool_shape(self):
        entry = ToolSpec(name="count", description="Count").to_openai_tool()
        assert entry["type"] == "function"
        assert entry["function"]["name"] == "count"
        assert entry["function"]["parameters"] == {"type": "object", "properties": {}}


class TestToolCall:
    def test_from_flat_dict_generates_id(self):
        call = ToolCall.from_dict({"name": "count", "arguments": {"value": 1}})
        assert call.name == "count"
        assert call.arguments == {"value": 1}
        assert call.id  # synthesised when the provider omits one

    def test_from_openai_dict_decodes_argument_string(self):
        call = ToolCall.from_dict(
            {
                "id": "abc",
                "type": "function",
                "function": {"name": "count", "arguments": '{"value": 2}'},
            }
        )
        assert call.id == "abc"
        assert call.name == "count"
        assert call.arguments == {"value": 2}

    def test_from_openai_dict_handles_unparseable_arguments(self):
        call = ToolCall.from_dict(
            {"id": "x", "function": {"name": "t", "arguments": "not json"}}
        )
        assert call.arguments == {"input": "not json"}

    def test_from_dict_wraps_non_mapping_arguments(self):
        call = ToolCall.from_dict({"id": "1", "name": "t", "arguments": [1, 2]})
        assert call.arguments == {"input": [1, 2]}

    def test_to_openai_dict_encodes_arguments(self):
        payload = ToolCall(id="1", name="t", arguments={"a": 1}).to_openai_dict()
        assert payload["function"]["arguments"] == '{"a": 1}'

    def test_to_dict_round_trip(self):
        call = ToolCall(id="1", name="t", arguments={"a": 1})
        assert ToolCall.from_dict(call.to_dict()).to_dict() == call.to_dict()


class TestMessage:
    def test_role_string_is_coerced(self):
        assert Message(role="user", content="hi").role is Role.USER

    def test_constructors(self):
        assert Message.system("s").role is Role.SYSTEM
        assert Message.user("u").content == "u"
        assistant = Message.assistant("thinking", tool_calls=[ToolCall(id="1", name="t")])
        assert assistant.tool_calls[0].name == "t"
        assert Message.tool("obs", tool_call_id="1").role is Role.TOOL

    def test_to_dict_includes_tool_calls(self):
        message = Message.assistant("", tool_calls=[ToolCall(id="1", name="t")])
        payload = message.to_dict()
        assert payload["role"] == "assistant"
        assert payload["tool_calls"] == [{"id": "1", "name": "t", "arguments": {}}]

    def test_to_dict_omits_absent_optional_fields(self):
        payload = Message.user("hi").to_dict()
        assert "name" not in payload
        assert "tool_call_id" not in payload

    def test_to_openai_dict_for_tool_role(self):
        payload = Message.tool("obs", tool_call_id="1", name="t").to_openai_dict()
        assert payload == {
            "role": "tool",
            "content": "obs",
            "tool_call_id": "1",
            "name": "t",
        }

    def test_to_openai_dict_encodes_nested_tool_calls(self):
        payload = Message.assistant(
            "thinking", tool_calls=[ToolCall(id="1", name="t", arguments={"a": 1})]
        ).to_openai_dict()
        assert payload["content"] == "thinking"
        assert payload["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


class TestToolResult:
    def test_failure_sets_error_content(self):
        result = ToolResult.failure("1", "t", "boom")
        assert result.ok is False
        assert result.error == "boom"
        assert result.content == "ERROR: boom"

    def test_to_dict_omits_absent_fields(self):
        payload = ToolResult(call_id="1", name="t", content="ok").to_dict()
        assert payload["ok"] is True
        assert "error" not in payload
        assert "data" not in payload

    def test_to_dict_includes_data_when_present(self):
        payload = ToolResult(call_id="1", name="t", data={"a": 1}).to_dict()
        assert payload["data"] == {"a": 1}


class TestAgentStep:
    def test_observation_joins_results_in_order(self):
        step = AgentStep(
            index=0,
            results=[
                ToolResult(call_id="1", name="a", content="one"),
                ToolResult(call_id="2", name="b", content="two"),
            ],
        )
        assert step.observation == "one\ntwo"
        assert step.failed is False

    def test_failed_is_true_when_any_result_failed(self):
        step = AgentStep(index=0, results=[ToolResult.failure("1", "a", "nope")])
        assert step.failed is True

    def test_total_tokens(self):
        assert AgentStep(index=0, prompt_tokens=3, completion_tokens=4).total_tokens == 7

    def test_to_dict_includes_nested_calls_and_results(self):
        step = AgentStep(
            index=1,
            thought="t",
            tool_calls=[ToolCall(id="1", name="tool")],
            results=[ToolResult(call_id="1", name="tool", content="ok")],
            status=StepStatus.SUCCEEDED,
        )
        payload = step.to_dict()
        assert payload["index"] == 1
        assert payload["status"] == "succeeded"
        assert payload["tool_calls"][0]["name"] == "tool"
        assert payload["results"][0]["content"] == "ok"


class TestAgentTrace:
    def _trace(self) -> AgentTrace:
        trace = AgentTrace(goal="count edges", started_at=10.0, finished_at=12.5)
        trace.steps.append(
            AgentStep(
                index=0,
                prompt_tokens=5,
                completion_tokens=2,
                tool_calls=[ToolCall(id="1", name="a")],
                results=[ToolResult(call_id="1", name="a", content="ok")],
                status=StepStatus.SUCCEEDED,
            )
        )
        trace.steps.append(
            AgentStep(
                index=1,
                prompt_tokens=7,
                completion_tokens=3,
                tool_calls=[ToolCall(id="2", name="b")],
                results=[ToolResult.failure("2", "b", "bad")],
                status=StepStatus.FAILED,
            )
        )
        trace.final_answer = "42"
        trace.status = StepStatus.SUCCEEDED
        return trace

    def test_aggregate_counts(self):
        trace = self._trace()
        assert trace.tool_call_count == 2
        assert trace.tool_error_count == 1
        assert trace.prompt_tokens == 12
        assert trace.completion_tokens == 5
        assert trace.total_tokens == 17

    def test_duration(self):
        assert self._trace().duration_s == pytest.approx(2.5)

    def test_duration_defaults_finished_to_started(self):
        trace = AgentTrace(goal="g", started_at=5.0)
        assert trace.duration_s == pytest.approx(0.0)

    def test_duration_none_without_start(self):
        assert AgentTrace(goal="g").duration_s is None

    def test_summary_mentions_status_and_counts(self):
        summary = self._trace().summary()
        assert "succeeded" in summary
        assert "tool_errors: 1" in summary
        assert "42" in summary

    def test_to_dict_shape(self):
        payload = self._trace().to_dict()
        assert payload["goal"] == "count edges"
        assert payload["tool_calls"] == 2
        assert payload["tool_errors"] == 1
        assert payload["total_tokens"] == 17
        assert payload["final_answer"] == "42"
        assert len(payload["steps"]) == 2


class TestAgentRunResult:
    def test_success_tracks_trace_status(self):
        trace = AgentTrace(goal="g", status=StepStatus.SUCCEEDED)
        assert AgentRunResult(trace=trace, answer="ok").success is True
        trace.status = StepStatus.FAILED
        assert AgentRunResult(trace=trace).success is False

    def test_str_prefers_answer_then_summary(self):
        trace = AgentTrace(goal="g", status=StepStatus.MAX_STEPS)
        assert str(AgentRunResult(trace=trace, answer="hi")) == "hi"
        assert "max_steps" in str(AgentRunResult(trace=trace))

    def test_to_dict_adds_answer_and_success(self):
        trace = AgentTrace(goal="g", status=StepStatus.SUCCEEDED)
        payload = AgentRunResult(trace=trace, answer="ok").to_dict()
        assert payload["answer"] == "ok"
        assert payload["success"] is True


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.max_steps == 8
        assert config.mode == "auto"

    @pytest.mark.parametrize("mode", ["native", "react", "auto"])
    def test_accepts_valid_modes(self, mode):
        assert AgentConfig(mode=mode).mode == mode

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            AgentConfig(mode="nope")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_steps": 0},
            {"max_tool_errors": -1},
            {"tool_retries": -1},
            {"max_tokens": 0},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            AgentConfig(**kwargs)
