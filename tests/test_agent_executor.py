"""Tests for the autonomous multi-step agent execution loop."""
from __future__ import annotations

import asyncio
from typing import Dict, List

import pytest

from astroml.agent.executor import (
    DEFAULT_SYSTEM_PROMPT,
    AgentExecutor,
    format_tool_catalogue,
)
from astroml.agent.llm import LLMProvider, LLMResponse, ScriptedLLM
from astroml.agent.memory import ConversationMemory
from astroml.agent.planner import TaskPlanner
from astroml.agent.tools import ToolRegistry
from astroml.agent.types import (
    AgentConfig,
    AgentStep,
    StepStatus,
    ToolCall,
    ToolSpec,
)


def _run(coro):
    """Execute a coroutine from a synchronous test."""
    return asyncio.run(coro)


def _make_registry():
    """Registry with a counting tool plus an always-failing tool."""
    calls: List = []

    def count_edges(start_ts: int = 0, end_ts: int = 10) -> Dict[str, int]:
        """Count the transactions in a timestamp window."""
        calls.append((start_ts, end_ts))
        return {"edges": 3}

    def fail_tool() -> None:
        """Always raises an error."""
        raise RuntimeError("boom")

    return ToolRegistry([count_edges, fail_tool]), calls


class TestFormatToolCatalogue:
    def test_lists_names_descriptions_and_arguments(self):
        text = format_tool_catalogue(
            [
                ToolSpec(
                    name="count",
                    description="Count things",
                    parameters={
                        "type": "object",
                        "properties": {"start_ts": {"type": "integer"}},
                    },
                )
            ]
        )
        assert "- count: Count things" in text
        assert "start_ts: integer" in text

    def test_handles_empty_registry(self):
        assert "No tools are available" in format_tool_catalogue([])


# ---------------------------------------------------------------------------
# Happy paths: reason -> act -> observe -> answer
# ---------------------------------------------------------------------------

class TestSuccessfulRuns:
    def test_react_loop_executes_tools_then_answers(self):
        registry, calls = _make_registry()
        llm = ScriptedLLM(
            [
                "Thought: count them\n"
                "Action: count_edges\n"
                'Action Input: {"start_ts": 1, "end_ts": 5}',
                "Thought: done\nFinal Answer: 3 edges",
            ]
        )
        agent = AgentExecutor(llm, registry, AgentConfig(mode="react", max_steps=4))
        result = agent.run("how many edges?")

        assert result.success is True
        assert result.answer == "3 edges"
        assert calls == [(1, 5)]
        assert len(result.trace.steps) == 2
        assert result.trace.tool_call_count == 1
        assert result.trace.steps[0].metadata["source"] == "react"
        assert result.trace.steps[0].status is StepStatus.SUCCEEDED

    def test_native_tool_calls_are_executed(self):
        registry, calls = _make_registry()
        llm = ScriptedLLM(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "count_edges",
                            "arguments": {"start_ts": 2, "end_ts": 3},
                        }
                    ],
                },
                {"content": "There are 3 edges."},
            ]
        )
        agent = AgentExecutor(llm, registry, AgentConfig(mode="native", max_steps=4))
        result = agent.run("count them")

        assert result.success is True
        assert result.answer == "There are 3 edges."
        assert calls == [(2, 3)]
        assert result.trace.steps[0].metadata["source"] == "native"

    def test_auto_mode_prefers_native_calls(self):
        registry, calls = _make_registry()
        llm = ScriptedLLM(
            [
                {
                    "content": "thinking",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "count_edges",
                            "arguments": {"start_ts": 7},
                        }
                    ],
                },
                "Final Answer: 3",
            ]
        )
        agent = AgentExecutor(llm, registry, AgentConfig(max_steps=3))
        result = agent.run("count them")

        assert calls == [(7, 10)]
        assert result.trace.steps[0].metadata["source"] == "native"
        assert result.trace.steps[0].thought == "thinking"


    def test_auto_mode_falls_back_to_react_text(self):
        registry, calls = _make_registry()
        llm = ScriptedLLM(
            [
                'Thought: use the tool\nAction: count_edges\n'
                'Action Input: {"start_ts": 9}',
                "Thought: ok\nFinal Answer: 3 edges",
            ]
        )
        agent = AgentExecutor(llm, registry, AgentConfig(max_steps=3))
        result = agent.run("count them")

        assert result.success is True
        assert result.trace.steps[0].metadata["source"] == "react"
        assert calls == [(9, 10)]

    def test_multiple_tool_calls_in_one_step(self):
        registry, calls = _make_registry()
        llm = ScriptedLLM(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "a", "name": "count_edges", "arguments": {}},
                        {"id": "b", "name": "count_edges", "arguments": {}},
                    ],
                },
                "Final Answer: done",
            ]
        )
        agent = AgentExecutor(llm, registry, AgentConfig(mode="native", max_steps=3))
        result = agent.run("count twice")

        assert len(result.trace.steps[0].tool_calls) == 2
        assert result.trace.tool_call_count == 2
        assert len(calls) == 2

    def test_trace_metadata_records_provider_and_tools(self):
        registry, _ = _make_registry()
        llm = ScriptedLLM(["Final Answer: ok"], name="scripted", model="test-model")
        result = AgentExecutor(llm, registry).run("hi")

        assert result.trace.metadata["provider"] == "scripted"
        assert result.trace.metadata["model"] == "test-model"
        assert result.trace.metadata["tools"] == ["count_edges", "fail_tool"]
        assert result.trace.duration_s is not None


# ---------------------------------------------------------------------------
# Failure handling and loop bounds
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_empty_goal_is_rejected(self):
        agent = AgentExecutor(ScriptedLLM(["x"]), ToolRegistry())
        with pytest.raises(ValueError):
            agent.run("   ")

    def test_max_steps_is_enforced(self):
        registry, _ = _make_registry()
        llm = ScriptedLLM(
            ["Action: count_edges\nAction Input: {}"],
            on_exhausted="repeat_last",
        )
        agent = AgentExecutor(llm, registry, AgentConfig(mode="react", max_steps=2))
        result = agent.run("loop forever")

        assert result.success is False
        assert result.trace.status is StepStatus.MAX_STEPS
        assert len(result.trace.steps) == 2
        assert "max_steps=2" in result.trace.error

    def test_llm_failures_are_recorded_instead_of_raised(self):
        class Boom(LLMProvider):
            name = "boom"

            async def complete(
                self,
                messages,
                *,
                tools=None,
                temperature=0.0,
                max_tokens=None,
            ):
                raise RuntimeError("network down")

        agent = AgentExecutor(Boom(), ToolRegistry(), AgentConfig(max_steps=2))
        result = agent.run("hi")

        assert result.success is False
        assert result.trace.status is StepStatus.FAILED
        assert "network down" in result.trace.error
        assert result.trace.steps[0].status is StepStatus.FAILED

    def test_empty_model_response_fails_the_run(self):
        agent = AgentExecutor(ScriptedLLM([""]), ToolRegistry())
        result = agent.run("hi")

        assert result.success is False
        assert "neither tool calls nor an answer" in result.trace.error

    def test_tool_errors_are_fed_back_to_the_model(self):
        registry, _ = _make_registry()
        llm = ScriptedLLM(
            [
                "Action: fail_tool\nAction Input: {}",
                "Thought: recovered\nFinal Answer: sorry about that",
            ]
        )
        agent = AgentExecutor(llm, registry, AgentConfig(mode="react", max_steps=4))
        result = agent.run("try it")

        assert result.success is True
        assert result.trace.tool_error_count == 1
        second_prompt = llm.calls[1]
        assert any(
            message.role.value == "tool" and "ERROR: boom" in message.content
            for message in second_prompt
        )

    def test_unknown_tool_is_reported_back_to_the_model(self):
        llm = ScriptedLLM(
            [
                "Action: ghost\nAction Input: {}",
                "Thought: oops\nFinal Answer: unavailable",
            ]
        )
        agent = AgentExecutor(
            llm, ToolRegistry(), AgentConfig(mode="react", max_steps=3)
        )
        result = agent.run("hi")

        assert result.trace.tool_error_count == 1
        assert result.success is True

    def test_tool_error_budget_is_enforced(self):
        registry, _ = _make_registry()
        llm = ScriptedLLM(
            ["Action: fail_tool\nAction Input: {}"],
            on_exhausted="repeat_last",
        )
        agent = AgentExecutor(
            llm,
            registry,
            AgentConfig(mode="react", max_steps=5, max_tool_errors=1),
        )
        result = agent.run("try it")

        assert result.success is False
        assert result.trace.status is StepStatus.FAILED
        assert "max_tool_errors=1" in result.trace.error
        assert len(result.trace.steps) == 2

    def test_stop_on_tool_error_aborts_immediately(self):
        registry, _ = _make_registry()
        llm = ScriptedLLM(
            ["Action: fail_tool\nAction Input: {}", "Final Answer: unreachable"]
        )
        agent = AgentExecutor(
            llm,
            registry,
            AgentConfig(mode="react", max_steps=5, stop_on_tool_error=True),
        )
        result = agent.run("try it")

        assert result.trace.status is StepStatus.FAILED
        assert len(result.trace.steps) == 1
        assert "boom" in result.trace.error


# ---------------------------------------------------------------------------
# Prompt construction, callbacks, memory and planning
# ---------------------------------------------------------------------------

class TestAgentBehaviour:
    def test_system_prompt_lists_tools_and_react_protocol(self):
        registry, _ = _make_registry()
        agent = AgentExecutor(ScriptedLLM(["x"]), registry)
        text = agent.build_system_message().content

        assert DEFAULT_SYSTEM_PROMPT in text
        assert "count_edges" in text
        assert "Action Input" in text

    def test_native_mode_prompt_omits_react_protocol(self):
        agent = AgentExecutor(
            ScriptedLLM(["x"]), ToolRegistry(), AgentConfig(mode="native")
        )
        assert "Action Input" not in agent.build_system_message().content

    def test_tool_catalogue_can_be_disabled(self):
        registry, _ = _make_registry()
        agent = AgentExecutor(
            ScriptedLLM(["x"]),
            registry,
            AgentConfig(include_tool_specs_in_prompt=False),
        )
        assert "Available tools" not in agent.build_system_message().content

    def test_tool_allow_list_restricts_visible_tools(self):
        registry, _ = _make_registry()
        agent = AgentExecutor(
            ScriptedLLM(["Final Answer: ok"]),
            registry,
            tool_names=["count_edges"],
        )
        assert agent.tools.names() == ["count_edges"]
        assert [spec.name for spec in agent.tool_specs] == ["count_edges"]

    def test_on_step_callback_receives_completed_steps(self):
        seen: List[AgentStep] = []
        agent = AgentExecutor(
            ScriptedLLM(["Final Answer: done"]),
            ToolRegistry(),
            on_step=seen.append,
        )
        agent.run("hi")
        assert [step.index for step in seen] == [0]
        assert seen[0].status is StepStatus.SUCCEEDED

    def test_callback_exceptions_do_not_break_the_run(self):
        def explode(step: AgentStep) -> None:
            raise RuntimeError("callback boom")

        agent = AgentExecutor(
            ScriptedLLM(["Final Answer: done"]), ToolRegistry(), on_step=explode
        )
        assert agent.run("hi").success is True

    def test_memory_receives_system_goal_and_answer(self):
        memory = ConversationMemory()
        agent = AgentExecutor(
            ScriptedLLM(["Final Answer: done"]), ToolRegistry(), memory=memory
        )
        agent.run("hi")
        assert [message.role.value for message in memory.messages()] == [
            "system",
            "user",
            "assistant",
        ]

    def test_planner_pre_decomposes_the_goal(self):
        registry, _ = _make_registry()
        plan_json = '{"steps": [{"description": "count", "tool": "count_edges"}]}'
        llm = ScriptedLLM([plan_json, "Final Answer: 3 edges"])
        agent = AgentExecutor(
            llm,
            registry,
            AgentConfig(max_steps=3),
            planner=TaskPlanner(llm),
        )
        result = agent.run("count edges")

        assert result.success is True
        assert result.trace.metadata["plan"]["steps"][0]["description"] == "count"
        execution_prompt = llm.calls[1]
        user_message = next(
            message for message in execution_prompt if message.role.value == "user"
        )
        assert "Suggested plan" in user_message.content

    def test_planner_failures_are_tolerated(self):
        class BrokenPlanner:
            async def aplan(self, goal, *, context=None, tool_names=None):
                raise RuntimeError("planner down")

        agent = AgentExecutor(
            ScriptedLLM(["Final Answer: ok"]),
            ToolRegistry(),
            planner=BrokenPlanner(),
        )
        assert agent.run("hi").success is True

    def test_interpret_response_prefers_native_calls(self):
        agent = AgentExecutor(ScriptedLLM(["x"]), ToolRegistry())
        turn = agent.interpret_response(
            LLMResponse(content="thinking", tool_calls=[ToolCall(id="1", name="t")])
        )
        assert turn.calls[0].name == "t"
        assert turn.final_answer is None
        assert turn.source == "native"

    def test_arun_can_be_awaited_directly(self):
        agent = AgentExecutor(ScriptedLLM(["Final Answer: async"]), ToolRegistry())
        assert _run(agent.arun("hi")).answer == "async"

    def test_result_is_json_serialisable(self):
        registry, _ = _make_registry()
        agent = AgentExecutor(
            ScriptedLLM(
                [
                    "Action: count_edges\nAction Input: {}",
                    "Final Answer: 3 edges",
                ]
            ),
            registry,
            AgentConfig(max_steps=3),
        )
        payload = agent.run("count").to_dict()
        assert payload["answer"] == "3 edges"
        assert payload["success"] is True
        assert payload["steps"][0]["tool_calls"][0]["name"] == "count_edges"
