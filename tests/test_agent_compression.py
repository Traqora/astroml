"""Tests for prompt compression and token budgeting in the agent framework."""
from __future__ import annotations

import json
from typing import List

import pytest

from astroml.agent.compression import (
    CompressedPrompt,
    CompressionConfig,
    CompressionStats,
    MessageGroup,
    ObservationDedupe,
    PromptCompressor,
    ToolOutputCompressor,
    WhitespaceNormalizer,
    compress_text,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    extractive_summary,
    group_messages,
    normalize_whitespace,
    total_stats,
    truncate_text,
)
from astroml.agent.cli import _build_compressor, _build_parser, main
from astroml.agent.executor import AgentExecutor, format_tool_catalogue
from astroml.agent.llm import ScriptedLLM
from astroml.agent.tools import ToolRegistry
from astroml.agent.types import AgentConfig, Message, Role, ToolCall, ToolSpec


def _tool_exchange(payload: str, *, call_id: str = "call_1") -> List[Message]:
    """A system/goal/tool-call/observation slice sharing one big payload."""
    return [
        Message.system("You are a careful agent."),
        Message.user("Report the account totals."),
        Message.assistant(
            "",
            tool_calls=[ToolCall(id=call_id, name="account_features", arguments={})],
        ),
        Message.tool(payload, tool_call_id=call_id, name="account_features"),
    ]


def _assert_pairing(messages: List[Message]) -> None:
    """Every tool observation must stay attached to its assistant request."""
    pending: set = set()
    for message in messages:
        if message.role is Role.ASSISTANT:
            pending = {call.id for call in message.tool_calls}
            continue
        if message.role is Role.TOOL:
            assert pending, f"orphaned tool message: {message.content[:40]!r}"
            assert message.tool_call_id in pending, "observation split from request"
            pending.discard(message.tool_call_id)
            continue
        assert not pending, "assistant tool request lost its observations"


class TestTokenEstimation:
    def test_empty_text_costs_nothing(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_uses_the_four_characters_per_token_heuristic(self):
        assert estimate_tokens("a" * 400) == 100
        assert estimate_tokens("a" * 40) == 10

    def test_grows_with_content(self):
        assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)

    def test_message_cost_includes_role_overhead(self):
        assert estimate_message_tokens(Message.system("a" * 400)) == 104

    def test_message_cost_counts_tool_calls(self):
        message = Message.assistant(
            "a" * 400,
            tool_calls=[ToolCall(id="c1", name="count", arguments={"a": 1})],
        )
        assert estimate_message_tokens(message) > estimate_message_tokens(
            Message.assistant("a" * 400)
        )

    def test_prompt_cost_sums_messages(self):
        messages = [Message.system("a" * 400), Message.user("b" * 400)]
        assert estimate_messages_tokens(messages) == sum(
            estimate_message_tokens(message) for message in messages
        )

    def test_accepts_a_custom_estimator(self):
        assert estimate_tokens.__module__ == "astroml.agent.compression"
        assert estimate_messages_tokens([], estimator=lambda text: 7) == 0


class TestTruncateText:
    def test_short_text_is_untouched(self):
        assert truncate_text("hello", 10) == "hello"

    def test_zero_budget_empties_the_text(self):
        assert truncate_text("hello", 0) == ""

    def test_keeps_head_and_tail_with_an_explicit_marker(self):
        original = "x" * 400
        out = truncate_text(original, 20)
        assert len(out) < len(original)
        assert "characters elided" in out
        assert out.startswith("x" * 50)
        assert out.endswith("x" * 24)


class TestCompressText:
    def test_short_text_is_untouched(self):
        assert compress_text("short", limit=50) == "short"

    def test_json_payloads_are_shrunk_structurally(self):
        payload = json.dumps({"rows": list(range(100)), "note": "n" * 30})
        out = compress_text(payload, limit=60)
        data = json.loads(out)
        assert len(data["rows"]) == 21
        assert data["rows"][-1] == "... (+80 more items)"
        assert data["note"] == "n" * 30

    def test_long_strings_fall_back_to_elision(self):
        out = compress_text("z" * 400, limit=20)
        assert len(out) < 400
        assert "characters elided" in out


class TestWhitespaceNormalizer:
    def test_collapses_blank_runs_and_trailing_spaces(self):
        assert normalize_whitespace("line\r\n\r\n\r\n\r\nnext  \n") == "line\n\nnext"
        assert normalize_whitespace("a   \nb") == "a\nb"

    def test_preserves_significant_indentation(self):
        assert normalize_whitespace("    code()") == "    code()"

    def test_strategy_rewrites_messages_without_dropping_them(self):
        messages = [Message.user("hello  \n\n\n\nworld")]
        out = WhitespaceNormalizer().compress(messages, config=CompressionConfig())
        assert len(out) == 1
        assert out[0].content == "hello\n\nworld"
        assert messages[0].content == "hello  \n\n\n\nworld"


class TestToolOutputCompressor:
    def test_shrinks_long_observations(self):
        messages = [Message.tool("x" * 4000, tool_call_id="c1", name="t")]
        out = ToolOutputCompressor().compress(
            messages, config=CompressionConfig(tool_output_limit=100)
        )
        assert len(out) == 1
        assert len(out[0].content) < 4000
        assert estimate_tokens(out[0].content) < 200
        assert len(messages[0].content) == 4000

    def test_leaves_short_observations_alone(self):
        messages = [Message.tool("tiny", tool_call_id="c1", name="t")]
        out = ToolOutputCompressor().compress(messages, config=CompressionConfig())
        assert out[0] is messages[0]

    def test_never_touches_non_observation_roles(self):
        messages = [Message.user("y" * 4000)]
        out = ToolOutputCompressor().compress(
            messages, config=CompressionConfig(tool_output_limit=10)
        )
        assert out[0] is messages[0]


class TestObservationDedupe:
    def test_replaces_repeated_observations_with_a_pointer(self):
        payload = "row " * 500
        messages = [
            Message.tool(payload, tool_call_id="c1", name="acct"),
            Message.tool(payload, tool_call_id="c2", name="acct"),
        ]
        out = ObservationDedupe().compress(messages, config=CompressionConfig())
        assert out[0].content == payload
        assert "duplicate of" in out[1].content
        assert "acct" in out[1].content
        assert len(out) == 2
        assert messages[1].content == payload

    def test_distinct_observations_are_kept(self):
        messages = [
            Message.tool("first result", tool_call_id="c1", name="a"),
            Message.tool("second result", tool_call_id="c2", name="a"),
        ]
        out = ObservationDedupe().compress(messages, config=CompressionConfig())
        assert [item.content for item in out] == ["first result", "second result"]


class TestExtractiveSummary:
    def test_digests_turns_and_skips_system_messages(self):
        messages = _tool_exchange("rows of data") + [Message.assistant("done")]
        digest = extractive_summary(messages)
        assert digest.startswith("Earlier steps condensed to save tokens:")
        assert "You are a careful agent" not in digest
        assert "Report the account totals" in digest
        assert "account_features" in digest
        assert "assistant: done" in digest

    def test_collapses_consecutive_duplicates(self):
        messages = [
            Message.tool("same", tool_call_id="c1", name="t"),
            Message.tool("same", tool_call_id="c2", name="t"),
        ]
        digest = extractive_summary(messages)
        assert digest.count("- t -> same") == 1

    def test_returns_empty_when_there_is_nothing_to_summarise(self):
        assert extractive_summary([Message.system("only instructions")]) == ""

    def test_reports_how_many_items_it_omitted(self):
        messages = [Message.user(f"step {index}") for index in range(20)]
        digest = extractive_summary(messages, max_items=5)
        assert "15 older item(s) omitted" in digest
        assert "step 19" in digest
        assert "step 0" not in digest


class TestGroupMessages:
    def test_keeps_tool_requests_and_observations_together(self):
        messages = _tool_exchange("payload") + [Message.assistant("done")]
        groups = group_messages(messages)

        assert len(groups) == 4
        assert [group.role for group in groups] == [
            Role.SYSTEM,
            Role.USER,
            Role.ASSISTANT,
            Role.ASSISTANT,
        ]
        assert groups[2].is_tool_exchange is True
        assert len(groups[2].messages) == 2
        assert groups[0].is_system is True

    def test_all_observations_of_one_request_share_a_group(self):
        messages = [
            Message.assistant(
                "",
                tool_calls=[
                    ToolCall(id="c1", name="a", arguments={}),
                    ToolCall(id="c2", name="b", arguments={}),
                ],
            ),
            Message.tool("one", tool_call_id="c1", name="a"),
            Message.tool("two", tool_call_id="c2", name="b"),
        ]
        groups = group_messages(messages)
        assert len(groups) == 1
        assert len(groups[0].messages) == 3

    def test_an_unanswered_tool_message_stands_alone(self):
        groups = group_messages([Message.tool("orphan", tool_call_id="x", name="t")])
        assert len(groups) == 1
        assert groups[0].is_tool_exchange is False

    def test_group_helpers_report_tokens_and_flatten(self):
        groups = group_messages(_tool_exchange("payload"))
        assert isinstance(groups[2], MessageGroup)
        assert groups[2].tokens() > 0
        assert len(groups[2].flatten()) == 2


def _long_transcript() -> List[Message]:
    """System + goal + six tool exchanges + a final answer (15 messages)."""
    messages = [
        Message.system("You are a careful agent."),
        Message.user("Report the account totals."),
    ]
    for index in range(6):
        call_id = f"call_{index}"
        messages.append(
            Message.assistant(
                "",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name="account_features",
                        arguments={"index": index},
                    )
                ],
            )
        )
        messages.append(
            Message.tool(
                f"observation {index}: " + "row " * 400,
                tool_call_id=call_id,
                name="account_features",
            )
        )
    messages.append(Message.assistant("All accounts processed."))
    return messages


def _joined(messages: List[Message]) -> str:
    return "\n".join(message.content for message in messages)


class TestPromptCompressor:
    def test_never_mutates_the_supplied_transcript(self):
        messages = _long_transcript()
        snapshot = [message.content for message in messages]

        PromptCompressor().compress(messages)

        assert [message.content for message in messages] == snapshot

    def test_short_prompts_pass_through_unchanged(self):
        messages = _tool_exchange("tiny") + [Message.assistant("done")]
        out = PromptCompressor().compress(messages)

        assert len(out.messages) == len(messages)
        assert [message.content for message in out.messages] == [
            message.content for message in messages
        ]
        assert out.stats.saved_tokens == 0

    def test_empty_prompts_are_handled(self):
        out = PromptCompressor().compress([])
        assert out.messages == []
        assert out.stats.before_tokens == 0

    def test_replaces_the_old_middle_with_a_digest(self):
        out = PromptCompressor(
            config=CompressionConfig(keep_recent=2)
        ).compress(_long_transcript())
        text = _joined(out.messages)

        assert "You are a careful agent" in text
        assert "Report the account totals" in text
        assert "Earlier steps condensed to save tokens:" in text
        assert "observation 5" in text
        assert not any(
            message.content.startswith("observation 0")
            for message in out.messages
        ), "the raw old observation should have been replaced by the digest"
        assert out.stats.summarized_messages == 10
        assert "summary" in out.stats.strategies

        digest = next(
            message
            for message in out.messages
            if message.content.startswith("Earlier steps condensed")
        )
        assert "observation 0" in digest.content
        assert "\u2026" in digest.content
        # 15 messages - 10 summarised + the 1 digest message that replaces them.
        assert len(out.messages) == 6

    def test_keeps_every_tool_request_with_its_observations(self):
        for config in (
            CompressionConfig(),
            CompressionConfig(keep_recent=1),
            CompressionConfig(max_prompt_tokens=900, keep_recent=2),
            CompressionConfig(tool_output_limit=50, tool_output_hard_limit=10),
        ):
            out = PromptCompressor(config=config).compress(_long_transcript())
            _assert_pairing(out.messages)

    def test_budget_is_enforced_and_pinned_context_survives(self):
        messages = _long_transcript()
        kept = [messages[0], messages[1], messages[12], messages[13], messages[14]]
        budget = estimate_messages_tokens(kept) + 60

        out = PromptCompressor(
            config=CompressionConfig(keep_recent=2, max_prompt_tokens=budget)
        ).compress(messages)

        assert out.stats.after_tokens <= budget
        text = _joined(out.messages)
        assert "You are a careful agent" in text
        assert "Report the account totals" in text
        assert "observation 5" in text
        # The pinned recent window survives verbatim; older raw observations only
        # survive as one-line mentions inside the digest.
        assert not any(
            message.content.startswith("observation 4")
            for message in out.messages
        )
        assert "- account_features -> observation 4:" in text

    def test_older_observations_are_shrunk_before_anything_is_dropped(self):
        out = PromptCompressor(
            config=CompressionConfig(
                keep_recent=1,
                summarize_older=False,
                max_prompt_tokens=1200,
            )
        ).compress(_long_transcript())

        assert out.stats.after_tokens <= 1200
        assert "tool-output-hard" in out.stats.strategies

    def test_min_savings_threshold_returns_the_original_prompt(self):
        messages = _long_transcript()
        out = PromptCompressor(
            config=CompressionConfig(min_savings_tokens=1_000_000)
        ).compress(messages)

        assert out.messages[0] is messages[0]
        assert out.stats.after_tokens == out.stats.before_tokens

    def test_custom_summarizer_is_used(self):
        out = PromptCompressor(
            config=CompressionConfig(keep_recent=2),
            summarizer=lambda messages, config: "OLD WORK DIGEST",
        ).compress(_long_transcript())

        assert "OLD WORK DIGEST" in _joined(out.messages)
        assert out.stats.summarized_messages > 0

    def test_records_savings_and_strategies(self):
        # tool_output_limit is set below the observation size on purpose, so both
        # the message-level and the digest-level strategies fire.
        out = PromptCompressor(
            config=CompressionConfig(tool_output_limit=200)
        ).compress(_long_transcript())

        assert isinstance(out, CompressedPrompt)
        assert out.stats.saved_tokens > 0
        assert out.stats.savings > 0
        assert (
            out.stats.before_messages
            == len(out.messages) + out.stats.dropped_messages
        )
        assert "tool-output" in out.stats.strategies
        assert "summary" in out.stats.strategies

    def test_default_tool_output_limit_leaves_moderate_observations_alone(self):
        out = PromptCompressor().compress(_long_transcript())

        assert "tool-output" not in out.stats.strategies
        assert "summary" in out.stats.strategies

    def test_callable_alias_matches_compress(self):
        compressor = PromptCompressor()
        messages = _long_transcript()
        assert compressor(messages).messages == compressor.compress(messages).messages

    def test_repr_names_the_active_strategies(self):
        text = repr(PromptCompressor())
        assert "PromptCompressor" in text
        assert "tool-output" in text


class TestCompressionConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_prompt_tokens": 0},
            {"max_prompt_tokens": 100, "reserve_output_tokens": 100},
            {"keep_recent": -1},
            {"tool_output_limit": 0},
            {"json_max_items": 0},
            {"summary_max_items": 0},
            {"min_savings_tokens": -1},
        ],
    )
    def test_invalid_tunables_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            CompressionConfig(**kwargs)

    def test_prompt_budget_accounts_for_reserved_reply_tokens(self):
        assert CompressionConfig().prompt_budget is None
        assert (
            CompressionConfig(
                max_prompt_tokens=1000, reserve_output_tokens=250
            ).prompt_budget
            == 750
        )


class TestTotalStats:
    def test_sums_every_field_and_unions_strategies(self):
        combined = total_stats(
            [
                CompressionStats(
                    before_tokens=100,
                    after_tokens=60,
                    before_messages=5,
                    after_messages=4,
                    dropped_messages=1,
                    summarized_messages=2,
                    strategies=["whitespace"],
                ),
                CompressionStats(
                    before_tokens=80,
                    after_tokens=50,
                    before_messages=4,
                    after_messages=3,
                    dropped_messages=1,
                    strategies=["whitespace", "tool-output"],
                ),
            ]
        )
        assert (combined.before_tokens, combined.after_tokens) == (180, 110)
        assert combined.saved_tokens == 70
        assert (combined.before_messages, combined.after_messages) == (9, 7)
        assert combined.dropped_messages == 2
        assert combined.summarized_messages == 2
        assert combined.strategies == ["whitespace", "tool-output"]

    def test_empty_input_is_a_no_op(self):
        assert total_stats([]).before_tokens == 0


# ---------------------------------------------------------------------------
# Integration: executor and CLI
# ---------------------------------------------------------------------------


def _big_registry(payload: str) -> ToolRegistry:
    def account_features(limit: int = 5) -> str:
        """Return a deliberately large observation."""
        return payload

    return ToolRegistry([account_features])


def _tool_call_llm() -> ScriptedLLM:
    return ScriptedLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "account_features", "arguments": {}}
                ],
            },
            "Final Answer: done",
        ]
    )


class TestExecutorCompression:
    def _execute(self, *, compressor=None, payload: str = "row " * 5000):
        llm = _tool_call_llm()
        agent = AgentExecutor(
            llm,
            _big_registry(payload),
            AgentConfig(max_steps=3),
            compressor=compressor,
        )
        return agent, agent.run("summarise the rows"), llm

    def test_provider_sees_a_smaller_prompt_than_memory(self):
        agent, result, llm = self._execute(
            compressor=PromptCompressor(
                config=CompressionConfig(tool_output_limit=200)
            )
        )

        assert result.success is True
        assert len(llm.calls) == 2

        sent = next(m for m in llm.calls[1] if m.role is Role.TOOL)
        stored = next(m for m in agent.memory.messages() if m.role is Role.TOOL)
        assert len(sent.content) < len(stored.content)
        assert len(stored.content) > 3000

    def test_trace_records_per_step_and_aggregate_stats(self):
        _, result, _ = self._execute(
            compressor=PromptCompressor(
                config=CompressionConfig(tool_output_limit=200)
            )
        )

        totals = result.trace.metadata["compression"]
        assert totals["enabled"] is True
        assert totals["turns"] == 2
        assert totals["saved_tokens"] > 0
        assert totals["before_tokens"] > totals["after_tokens"]
        assert "tool-output" in totals["strategies"]

        per_step = result.trace.steps[0].metadata["compression"]
        assert per_step["before_tokens"] > 0

    def test_no_compression_metadata_without_a_compressor(self):
        agent, result, llm = self._execute()

        assert result.success is True
        assert "compression" not in result.trace.metadata
        assert "compression" not in result.trace.steps[0].metadata
        sent = next(m for m in llm.calls[1] if m.role is Role.TOOL)
        assert len(sent.content) > 3000

    def test_memory_is_never_rewritten(self):
        agent, result, _ = self._execute(
            compressor=PromptCompressor(
                config=CompressionConfig(keep_recent=1, tool_output_limit=100)
            )
        )
        assert result.success is True
        assert len(agent.memory.messages()) >= 5


class TestCompactToolCatalogue:
    def _spec(self) -> ToolSpec:
        return ToolSpec(
            name="count",
            description="Count things",
            parameters={
                "type": "object",
                "properties": {"start_ts": {"type": "integer"}},
            },
        )

    def test_compact_layout_is_one_line_per_tool(self):
        compact = format_tool_catalogue([self._spec()], compact=True)
        full = format_tool_catalogue([self._spec()])

        assert "- count(start_ts: integer): Count things" in compact
        assert "arguments:" not in compact
        assert len(compact) < len(full)
        assert "count" in full and "arguments:" in full

    def test_config_flag_reaches_the_system_prompt(self):
        agent = AgentExecutor(
            ScriptedLLM([]),
            _big_registry("x"),
            AgentConfig(compact_tool_catalogue=True),
        )
        text = agent.build_system_message().content
        assert "- account_features(limit: integer):" in text
        assert "arguments:" not in text


class TestCompressionFlags:
    def _args(self, *flags: str):
        return _build_parser().parse_args([*flags, "hi"])

    def test_compression_is_off_by_default(self):
        assert _build_compressor(self._args()) is None

    def test_budget_flags_imply_compression(self):
        compressor = _build_compressor(self._args("--max-prompt-tokens", "4000"))
        assert isinstance(compressor, PromptCompressor)
        assert compressor.config.max_prompt_tokens == 4000

    def test_keep_recent_and_tool_limit_are_forwarded(self):
        compressor = _build_compressor(
            self._args(
                "--compress",
                "--compress-keep-recent",
                "2",
                "--tool-output-limit",
                "150",
            )
        )
        assert compressor.config.keep_recent == 2
        assert compressor.config.tool_output_limit == 150

    def test_unspecified_tunables_keep_the_defaults(self):
        compressor = _build_compressor(self._args("--compress"))
        assert compressor.config.keep_recent == CompressionConfig().keep_recent
        assert compressor.config.tool_output_limit == (
            CompressionConfig().tool_output_limit
        )

    def test_compact_tools_flag_is_parsed(self):
        assert self._args("--compact-tools").compact_tools is True
        assert self._args().compact_tools is False

    def test_invalid_budget_is_a_configuration_error(self, capsys):
        assert main(["--provider", "echo", "--max-prompt-tokens", "0", "hi"]) == 2
        assert "error:" in capsys.readouterr().err

    def test_summary_is_printed_after_the_run(self, capsys):
        exit_code = main(["--provider", "echo", "--compress", "hello"])

        assert exit_code == 0
        assert "compression:" in capsys.readouterr().err

    def test_json_output_carries_the_aggregate_stats(self, capsys):
        exit_code = main(["--provider", "echo", "--compress", "--json", "hello"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        totals = payload["metadata"]["compression"]
        assert totals["enabled"] is True
        assert totals["turns"] >= 1

    def test_quiet_runs_still_succeed(self, capsys):
        exit_code = main(["--provider", "echo", "--compress", "--quiet", "hello"])

        assert exit_code == 0
        assert "compression:" not in capsys.readouterr().err
