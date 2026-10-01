"""Tests for ReAct parsing and LLM driven task decomposition."""
from __future__ import annotations

import asyncio

import pytest

from astroml.agent.llm import ScriptedLLM
from astroml.agent.planner import (
    Plan,
    PlanStep,
    ReActOutput,
    ReActParser,
    TaskPlanner,
    extract_json_block,
)


def _run(coro):
    """Execute a coroutine from a synchronous test."""
    return asyncio.run(coro)


class TestExtractJsonBlock:
    def test_parses_bare_json_object(self):
        assert extract_json_block('{"a": 1}') == {"a": 1}

    def test_parses_fenced_json(self):
        text = 'Here you go:\n```json\n{"steps": ["a"]}\n```\nThanks!'
        assert extract_json_block(text) == {"steps": ["a"]}

    def test_parses_json_embedded_in_prose(self):
        assert extract_json_block('Sure! {"steps": ["a"]} done') == {"steps": ["a"]}

    def test_parses_top_level_array(self):
        assert extract_json_block("[1, 2, 3]") == [1, 2, 3]

    def test_ignores_braces_inside_strings(self):
        assert extract_json_block('{"a": "}"}') == {"a": "}"}

    def test_returns_none_without_json(self):
        assert extract_json_block("no json here") is None
        assert extract_json_block("") is None

    def test_returns_none_for_malformed_json(self):
        assert extract_json_block("{broken: true}") is None


class TestReActParser:
    def test_parses_action_with_json_input(self):
        output = ReActParser.parse(
            "Thought: I need a window count\n"
            "Action: window_stats\n"
            'Action Input: {"start_ts": 1, "end_ts": 2}'
        )
        assert output.thought == "I need a window count"
        assert output.action == "window_stats"
        assert output.action_input == {"start_ts": 1, "end_ts": 2}
        assert output.has_action is True
        assert output.is_final is False

    def test_parses_multi_line_json_input(self):
        output = ReActParser.parse(
            'Action: window_stats\nAction Input: {\n  "start_ts": 1\n}'
        )
        assert output.action_input == {"start_ts": 1}

    def test_parses_key_value_input(self):
        output = ReActParser.parse("Action: window_stats\nAction Input: start_ts=1, end_ts=2")
        assert output.action_input == {"start_ts": 1, "end_ts": 2}

    def test_parses_free_text_input(self):
        output = ReActParser.parse("Action: describe\nAction Input: the whole graph")
        assert output.action_input == {"input": "the whole graph"}

    def test_parses_final_answer(self):
        output = ReActParser.parse("Thought: done\nFinal Answer: 3 edges")
        assert output.is_final is True
        assert output.final_answer == "3 edges"
        assert output.action is None

    def test_markers_are_case_insensitive(self):
        output = ReActParser.parse("thought: lower\naction: tool\naction input: {}")
        assert output.thought == "lower"
        assert output.action == "tool"

    def test_ignores_hallucinated_observation(self):
        output = ReActParser.parse(
            "Observation: 5 edges\nThought: got it\nFinal Answer: five"
        )
        assert output.final_answer == "five"
        assert "Observation" not in output.thought

    def test_plain_prose_is_treated_as_thought(self):
        output = ReActParser.parse("The graph has 12 accounts.")
        assert output.thought == "The graph has 12 accounts."
        assert output.action is None
        assert output.is_final is False

    def test_empty_input_returns_empty_output(self):
        output = ReActParser.parse("   ")
        assert output == ReActOutput()

    def test_marker_without_content_still_signals_final_answer(self):
        output = ReActParser.parse("Final Answer:")
        assert output.is_final is True
        assert output.final_answer == ""

    def test_to_tool_call_requires_an_action(self):
        with pytest.raises(ValueError):
            ReActOutput(thought="no action").to_tool_call()

    def test_to_tool_call_builds_call(self):
        output = ReActParser.parse('Action: count\nAction Input: {"value": 2}')
        call = output.to_tool_call(call_id="c1")
        assert call.id == "c1"
        assert call.name == "count"
        assert call.arguments == {"value": 2}


class TestPlanStep:
    def test_to_dict_includes_arguments_only_when_present(self):
        assert PlanStep(description="a").to_dict() == {"description": "a", "tool": None}
        with_arguments = PlanStep(description="a", tool="t", arguments={"x": 1})
        assert with_arguments.to_dict()["arguments"] == {"x": 1}


class TestPlan:
    def test_is_empty_and_len(self):
        plan = Plan(goal="g")
        assert plan.is_empty is True
        assert len(plan) == 0

    def test_iteration_yields_steps(self):
        plan = Plan(goal="g", steps=[PlanStep(description="a"), PlanStep(description="b")])
        assert [step.description for step in plan] == ["a", "b"]

    def test_to_dict(self):
        plan = Plan(goal="g", steps=[PlanStep(description="a")], notes="n")
        assert plan.to_dict() == {
            "goal": "g",
            "steps": [{"description": "a", "tool": None}],
            "notes": "n",
        }


class TestTaskPlanner:
    def _planner(self, script, **kwargs) -> TaskPlanner:
        return TaskPlanner(ScriptedLLM([script]), **kwargs)

    def test_parses_json_steps_from_model_output(self):
        plan = TaskPlanner.parse(
            '{"steps": [{"description": "load", "tool": "graph_overview", '
            '"arguments": {"limit": 1}}, "report"], "notes": "keep it short"}',
            goal="analyse",
        )
        assert plan.goal == "analyse"
        assert plan.notes == "keep it short"
        assert [step.description for step in plan.steps] == ["load", "report"]
        assert plan.steps[0].tool == "graph_overview"
        assert plan.steps[0].arguments == {"limit": 1}
        assert plan.steps[1].tool is None

    def test_accepts_plan_key_and_top_level_list(self):
        assert len(TaskPlanner.parse('{"plan": ["a", "b"]}')) == 2
        assert len(TaskPlanner.parse('[{"task": "a"}]')) == 1

    def test_parses_fenced_json(self):
        assert len(TaskPlanner.parse('```json\n{"steps": ["a"]}\n```')) == 1

    def test_truncates_to_max_steps(self):
        plan = TaskPlanner.parse('["a", "b", "c"]', max_steps=2)
        assert [step.description for step in plan.steps] == ["a", "b"]

    def test_unparseable_output_yields_empty_plan(self):
        plan = TaskPlanner.parse("I cannot help with that", goal="g")
        assert plan.is_empty is True
        assert plan.raw == "I cannot help with that"

    def test_blank_and_malformed_entries_are_skipped(self):
        plan = TaskPlanner.parse('["a", "", 3, {"description": "b"}]')
        assert [step.description for step in plan.steps] == ["a", "b"]

    def test_non_mapping_arguments_are_discarded(self):
        plan = TaskPlanner.parse('[{"description": "a", "arguments": "nope"}]')
        assert plan.steps[0].arguments == {}

    def test_rejects_invalid_max_steps(self):
        with pytest.raises(ValueError):
            TaskPlanner(ScriptedLLM([]), max_steps=0)

    def test_build_messages_includes_context_and_tools(self):
        planner = TaskPlanner(ScriptedLLM([]), max_steps=3)
        messages = planner.build_messages(
            "goal", context="edges: 2", tool_names=["graph_overview"]
        )
        assert messages[0].role.value == "system"
        user_text = messages[1].content
        assert "goal" in user_text
        assert "edges: 2" in user_text
        assert "graph_overview" in user_text
        assert "at most 3 steps" in user_text

    def test_aplan_returns_parsed_plan(self):
        planner = TaskPlanner(
            ScriptedLLM(['{"steps": [{"description": "one", "tool": "t"}]}'])
        )
        plan = _run(planner.aplan("do it"))
        assert plan.goal == "do it"
        assert plan.steps[0].tool == "t"

    def test_plan_sync_wrapper(self):
        assert len(self._planner('["a"]').plan("do it")) == 1
