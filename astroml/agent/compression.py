"""Prompt compression and token budgeting for agent runs.

An agent loop resends the whole transcript on every turn, so prompt cost grows
roughly quadratically with the number of reasoning steps.  This module lowers
that cost without hiding anything the model needs:

* :func:`estimate_tokens` — dependency free token approximation, so prompts can
  be budgeted without shipping a tokenizer.
* :class:`WhitespaceNormalizer`, :class:`ToolOutputCompressor` and
  :class:`ObservationDedupe` — message level rewrites that keep every message
  (and therefore every tool-call pairing) in place.
* :func:`extractive_summary` — condense old turns into a compact digest with no
  extra model call, so compression stays free and deterministic.
* :class:`PromptCompressor` — applies the strategies above, replaces the
  un-pinned middle of the transcript with a digest, and trims to a token budget
  when one is configured.

Compression only ever touches the messages *sent to the provider*: the
executor's :class:`~astroml.agent.memory.Memory` is never mutated, so traces and
audits still see the full, uncompressed conversation.

Correctness note: an assistant message that requests tools must stay grouped
with the ``tool`` messages answering it, because OpenAI compatible APIs reject
transcripts where the two are separated.  :func:`group_messages` models those
atomic units and every strategy preserves them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .types import Message, Role

#: Average characters per token for English prose and JSON (vendor heuristic).
CHARS_PER_TOKEN = 4.0
#: Average words per token for English prose.
WORDS_PER_TOKEN = 0.75
#: Per-message overhead (role markers, separators) added by chat providers.
MESSAGE_OVERHEAD_TOKENS = 4

_NOT_JSON: Any = object()


def estimate_tokens(text: Optional[str]) -> int:
    """Approximate the token count of *text* without a tokenizer.

    Uses the widely used ``~4 characters`` / ``~0.75 words`` per token
    heuristics and rounds up, so the estimate is deliberately conservative for
    budgeting purposes.  Swap in ``tiktoken`` or a provider tokenizer by passing
    your own ``estimator`` to :class:`PromptCompressor` when exact counts matter.
    """
    if not text:
        return 0
    content = str(text)
    estimate = max(
        len(content) / CHARS_PER_TOKEN,
        len(content.split()) / WORDS_PER_TOKEN,
    )
    return max(1, int(math.ceil(estimate)))


def estimate_message_tokens(
    message: Message,
    *,
    estimator: Callable[[str], int] = estimate_tokens,
) -> int:
    """Approximate the tokens one :class:`~astroml.agent.types.Message` costs."""
    total = MESSAGE_OVERHEAD_TOKENS + estimator(message.content)
    for call in message.tool_calls:
        total += MESSAGE_OVERHEAD_TOKENS + estimator(call.name)
        total += estimator(json.dumps(call.arguments, default=str))
    if message.name:
        total += estimator(message.name)
    return total


def estimate_messages_tokens(
    messages: Sequence[Message],
    *,
    estimator: Callable[[str], int] = estimate_tokens,
) -> int:
    """Approximate the tokens a whole prompt costs."""
    return sum(
        estimate_message_tokens(message, estimator=estimator) for message in messages
    )


def estimate_messages_tokens_parallel(
    messages: Sequence[Message],
    *,
    estimator: Callable[[str], int] = estimate_tokens,
    max_workers: Optional[int] = None,
) -> int:
    """Approximate the tokens a whole prompt costs using parallel processing.

    Uses ThreadPoolExecutor to parallelize token estimation across messages.
    Beneficial for large message counts where estimation becomes CPU-bound.

    Args:
        messages: Sequence of messages to estimate.
        estimator: Token estimation function.
        max_workers: Maximum number of worker threads. Defaults to CPU count.

    Returns:
        Total estimated tokens for all messages.
    """
    if not messages:
        return 0
    if len(messages) < 10:  # Threshold below which overhead outweighs benefits
        return estimate_messages_tokens(messages, estimator=estimator)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(estimate_message_tokens, message, estimator=estimator)
            for message in messages
        ]
        return sum(future.result() for future in as_completed(futures))


def truncate_text(
    text: str,
    max_tokens: int,
    *,
    head_ratio: float = 0.7,
    estimator: Callable[[str], int] = estimate_tokens,
) -> str:
    """Keep the head and tail of *text*, eliding the middle to fit *max_tokens*.

    Head/tail truncation keeps the most useful ends of an observation (the start
    of a payload and its trailing summary fields) and leaves an explicit elision
    marker, so the model knows information was dropped rather than silently
    inventing it.
    """
    if not text:
        return text
    if max_tokens <= 0:
        return ""
    if estimator(text) <= max_tokens:
        return text

    ratio = min(max(float(head_ratio), 0.0), 1.0)
    budget_chars = max(64, int(max_tokens * CHARS_PER_TOKEN))
    head_chars = max(1, int(budget_chars * ratio))
    tail_chars = max(0, budget_chars - head_chars)
    elided = len(text) - head_chars - tail_chars
    if elided <= 0:
        return text

    head = text[:head_chars].rstrip()
    marker = f"\n... [{elided} characters elided] ..."
    if not tail_chars:
        return f"{head}{marker}"
    return f"{head}{marker}\n{text[-tail_chars:].lstrip()}"


def _clip_text(text: str, limit: int) -> str:
    """Clip *text* to *limit* characters with an explicit remainder count."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (+{len(text) - limit} chars)"


def _loads_jsonish(text: str) -> Any:
    """Decode *text* as JSON, returning :data:`_NOT_JSON` when it is not JSON."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return _NOT_JSON
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _NOT_JSON


def _shrink_value(
    value: Any,
    *,
    items: int,
    string: int,
    depth: int = 0,
    max_depth: int = 6,
) -> Any:
    """Recursively clip long JSON values while preserving structure and keys."""
    if depth >= max_depth:
        if isinstance(value, str):
            return _clip_text(value, string)
        if isinstance(value, (Mapping, list, tuple)):
            return _clip_text(json.dumps(value, default=str), string)
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _shrink_value(
                item, items=items, string=string, depth=depth + 1, max_depth=max_depth
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        kept = [
            _shrink_value(
                item, items=items, string=string, depth=depth + 1, max_depth=max_depth
            )
            for item in value[:items]
        ]
        remaining = len(value) - len(kept)
        if remaining > 0:
            kept.append(f"... (+{remaining} more items)")
        return kept
    if isinstance(value, str):
        return _clip_text(value, string)
    return value


def compress_text(
    text: str,
    *,
    limit: int,
    json_max_items: int = 20,
    json_max_string: int = 200,
    estimator: Callable[[str], int] = estimate_tokens,
) -> str:
    """Shrink a single tool observation towards *limit* tokens.

    JSON payloads are shrunk **structurally** first — long arrays are truncated
    with an explicit ``+N more items`` marker and long strings are clipped — so
    the model keeps the schema and the most informative leading values instead
    of an arbitrary character prefix.  Only if that is not enough does plain
    head/tail elision kick in.
    """
    if not text or estimator(text) <= limit:
        return text

    payload = _loads_jsonish(text)
    if payload is not _NOT_JSON:
        rendered = json.dumps(
            _shrink_value(payload, items=json_max_items, string=json_max_string),
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if estimator(rendered) <= limit:
            return rendered
        text = rendered
    return truncate_text(text, limit, estimator=estimator)


def compress_text_batch(
    texts: Sequence[str],
    *,
    limit: int,
    json_max_items: int = 20,
    json_max_string: int = 200,
    estimator: Callable[[str], int] = estimate_tokens,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Compress multiple texts in parallel.

    Uses ThreadPoolExecutor to parallelize text compression across multiple
    observations. Beneficial for batch processing of tool outputs.

    Args:
        texts: Sequence of texts to compress.
        limit: Token limit for each text.
        json_max_items: Items kept from long JSON arrays.
        json_max_string: Characters kept from long JSON strings.
        estimator: Token estimation function.
        max_workers: Maximum number of worker threads.

    Returns:
        List of compressed texts in the same order as input.
    """
    if not texts:
        return []
    if len(texts) < 5:  # Threshold below which overhead outweighs benefits
        return [
            compress_text(
                text,
                limit=limit,
                json_max_items=json_max_items,
                json_max_string=json_max_string,
                estimator=estimator,
            )
            for text in texts
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                compress_text,
                text,
                limit=limit,
                json_max_items=json_max_items,
                json_max_string=json_max_string,
                estimator=estimator,
            ): idx
            for idx, text in enumerate(texts)
        }
        results = [None] * len(texts)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results


_TRAILING_WHITESPACE = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Remove whitespace that carries no information for a language model.

    Line endings are unified, trailing spaces and runs of blank lines are
    collapsed, and leading/trailing blank lines are dropped.  Indentation inside
    the payload is deliberately preserved because it is significant in code and
    in pretty printed JSON.
    """
    if not text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _TRAILING_WHITESPACE.sub("\n", normalized)
    normalized = _BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip("\n").rstrip(" \t")


def _snippet(text: Optional[str], limit: int = 160) -> str:
    """Collapse whitespace and clip *text* for digest lines."""
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: max(1, limit - 1)]}\u2026"


def _copy_message(message: Message, **changes: Any) -> Message:
    """Copy *message* so compression never mutates caller owned history."""
    cloned = replace(message, **changes)
    cloned.tool_calls = list(message.tool_calls)
    cloned.metadata = dict(message.metadata)
    return cloned


@dataclass
class CompressionConfig:
    """Tunables for :class:`PromptCompressor`.

    Args:
        max_prompt_tokens: Budget for the compressed prompt. ``None`` disables
            budget trimming (the other strategies still apply).
        reserve_output_tokens: Tokens held back for the model's reply when
            ``max_prompt_tokens`` describes a whole context window.
        keep_recent: Number of trailing message groups kept verbatim.
        keep_goal: Keep the first non-system group (the goal) verbatim.
        tool_output_limit: Token ceiling for a single tool observation.
        tool_output_hard_limit: Stricter ceiling applied only when even the
            trimmed prompt does not fit the budget.
        json_max_items: Items kept from long JSON arrays in observations.
        json_max_string: Characters kept from long JSON strings.
        summarize_older: Replace the un-pinned middle with a digest message.
        summary_max_items: Digest lines kept by :func:`extractive_summary`.
        min_savings_tokens: Return the original prompt unchanged unless
            compression saves at least this many tokens.
        parallel_threshold: Minimum number of messages required to enable
            parallel processing. Below this threshold, sequential processing
            is used to avoid overhead.
        max_workers: Maximum number of worker threads for parallel operations.
            ``None`` uses the CPU count.
    """

    max_prompt_tokens: Optional[int] = None
    reserve_output_tokens: int = 0
    keep_recent: int = 6
    keep_goal: bool = True
    tool_output_limit: int = 800
    tool_output_hard_limit: int = 200
    json_max_items: int = 20
    json_max_string: int = 200
    summarize_older: bool = True
    summary_max_items: int = 12
    min_savings_tokens: int = 0
    parallel_threshold: int = 10
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1 when provided")
        if self.reserve_output_tokens < 0:
            raise ValueError("reserve_output_tokens must be >= 0")
        if (
            self.max_prompt_tokens is not None
            and self.reserve_output_tokens >= self.max_prompt_tokens
        ):
            raise ValueError(
                "reserve_output_tokens must be smaller than max_prompt_tokens"
            )
        if self.keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        if self.tool_output_limit < 1:
            raise ValueError("tool_output_limit must be >= 1")
        if self.tool_output_hard_limit < 1:
            raise ValueError("tool_output_hard_limit must be >= 1")
        if self.json_max_items < 1:
            raise ValueError("json_max_items must be >= 1")
        if self.json_max_string < 1:
            raise ValueError("json_max_string must be >= 1")
        if self.summary_max_items < 1:
            raise ValueError("summary_max_items must be >= 1")
        if self.min_savings_tokens < 0:
            raise ValueError("min_savings_tokens must be >= 0")
        if self.parallel_threshold < 0:
            raise ValueError("parallel_threshold must be >= 0")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided")

    @property
    def prompt_budget(self) -> Optional[int]:
        """Tokens available to the prompt, after reserving reply headroom."""
        if self.max_prompt_tokens is None:
            return None
        return max(1, self.max_prompt_tokens - self.reserve_output_tokens)


@dataclass
class CompressionStats:
    """What one compression pass achieved."""

    before_tokens: int = 0
    after_tokens: int = 0
    before_messages: int = 0
    after_messages: int = 0
    dropped_messages: int = 0
    summarized_messages: int = 0
    strategies: List[str] = field(default_factory=list)

    @property
    def saved_tokens(self) -> int:
        """Tokens removed by compression (never negative)."""
        return max(0, int(self.before_tokens) - int(self.after_tokens))

    @property
    def savings(self) -> float:
        """Fraction of the original prompt removed, in ``[0, 1]``."""
        if self.before_tokens <= 0:
            return 0.0
        return self.saved_tokens / float(self.before_tokens)

    def to_dict(self) -> Dict[str, Any]:
        """JSON serialisable summary, suitable for run metadata."""
        return {
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "saved_tokens": self.saved_tokens,
            "savings": round(self.savings, 6),
            "before_messages": self.before_messages,
            "after_messages": self.after_messages,
            "dropped_messages": self.dropped_messages,
            "summarized_messages": self.summarized_messages,
            "strategies": list(self.strategies),
        }

    def __str__(self) -> str:
        if not self.strategies and self.before_tokens == self.after_tokens:
            return (
                f"prompt unchanged ({self.after_tokens} tokens, "
                f"{self.after_messages} messages)"
            )
        text = (
            f"prompt {self.before_tokens} -> {self.after_tokens} tokens "
            f"(-{self.savings:.1%}), {self.before_messages} -> "
            f"{self.after_messages} messages"
        )
        if self.strategies:
            text += f" [{', '.join(self.strategies)}]"
        return text


def total_stats(stats: Iterable[CompressionStats]) -> CompressionStats:
    """Combine per-turn :class:`CompressionStats` into one aggregate."""
    combined = CompressionStats()
    strategies: List[str] = []
    for item in stats:
        combined.before_tokens += item.before_tokens
        combined.after_tokens += item.after_tokens
        combined.before_messages += item.before_messages
        combined.after_messages += item.after_messages
        combined.dropped_messages += item.dropped_messages
        combined.summarized_messages += item.summarized_messages
        for name in item.strategies:
            if name not in strategies:
                strategies.append(name)
    combined.strategies = strategies
    return combined


@dataclass
class CompressedPrompt:
    """Result of :meth:`PromptCompressor.compress`."""

    messages: List[Message] = field(default_factory=list)
    stats: CompressionStats = field(default_factory=CompressionStats)

    @property
    def tokens(self) -> int:
        """Estimated tokens of the compressed prompt."""
        return self.stats.after_tokens

    def __len__(self) -> int:
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "stats": self.stats.to_dict(),
        }


class Compressor(ABC):
    """Strategy that rewrites prompt messages in place, never removing them.

    Implementations must return a *new* list and must not mutate the messages
    they are given: the executor's memory owns those objects.  Because these
    strategies only rewrite message content, they can never split an assistant
    tool request from the ``tool`` messages answering it.
    """

    #: Identifier recorded in :attr:`CompressionStats.strategies`.
    name: str = "compressor"

    @abstractmethod
    def compress(
        self,
        messages: Sequence[Message],
        *,
        config: CompressionConfig,
        estimator: Callable[[str], int] = estimate_tokens,
    ) -> List[Message]:
        """Return a rewritten copy of *messages*."""


class WhitespaceNormalizer(Compressor):
    """Cheap, information preserving whitespace cleanup.

    Unifies line endings, strips trailing spaces and collapses runs of blank
    lines.  Indentation is preserved so code and pretty printed JSON stay
    readable, which keeps decoding quality intact while shaving tokens off every
    message that carries formatting noise.
    """

    name = "whitespace"

    def compress(
        self,
        messages: Sequence[Message],
        *,
        config: CompressionConfig,
        estimator: Callable[[str], int] = estimate_tokens,
    ) -> List[Message]:
        normalized: List[Message] = []
        for message in messages:
            content = normalize_whitespace(message.content)
            if content == message.content:
                normalized.append(message)
            else:
                normalized.append(_copy_message(message, content=content))
        return normalized


class ToolOutputCompressor(Compressor):
    """Structure preserving shrinker for tool observations.

    Long ``tool`` messages dominate real transcripts (they carry JSON payloads),
    so they are shrunk first.  JSON is reduced structurally before any character
    level elision, which preserves keys, types and the leading values a model
    needs to keep reasoning.

    Args:
        roles: Message roles treated as observations (default: tool only).
        limit: Optional token ceiling that overrides
            :attr:`CompressionConfig.tool_output_limit`.
    """

    name = "tool-output"

    def __init__(
        self,
        *,
        roles: Sequence[Role] = (Role.TOOL,),
        limit: Optional[int] = None,
    ) -> None:
        self.roles = tuple(roles)
        self.limit = limit

    def compress(
        self,
        messages: Sequence[Message],
        *,
        config: CompressionConfig,
        estimator: Callable[[str], int] = estimate_tokens,
    ) -> List[Message]:
        limit = self.limit if self.limit is not None else config.tool_output_limit
        compressed: List[Message] = []
        for message in messages:
            if message.role not in self.roles or not message.content:
                compressed.append(message)
                continue
            shrunk = compress_text(
                message.content,
                limit=limit,
                json_max_items=config.json_max_items,
                json_max_string=config.json_max_string,
                estimator=estimator,
            )
            if shrunk == message.content:
                compressed.append(message)
            else:
                compressed.append(_copy_message(message, content=shrunk))
        return compressed


class ObservationDedupe(Compressor):
    """Replace repeated observations with a pointer to the first occurrence.

    Models frequently re-run a tool with the same arguments; resending an
    identical multi-kilobyte payload buys nothing.  Identical observations are
    therefore replaced by a short pointer, but only when the pointer is
    genuinely cheaper than the payload it replaces.
    """

    name = "dedupe"

    def compress(
        self,
        messages: Sequence[Message],
        *,
        config: CompressionConfig,
        estimator: Callable[[str], int] = estimate_tokens,
    ) -> List[Message]:
        seen: Dict[str, str] = {}
        deduped: List[Message] = []
        for message in messages:
            if message.role is not Role.TOOL or not message.content:
                deduped.append(message)
                continue

            digest = hashlib.sha1(
                message.content.encode("utf-8", "replace")
            ).hexdigest()[:12]
            previous = seen.get(digest)
            if previous is None:
                label = message.tool_call_id or message.name or f"#{len(seen) + 1}"
                seen[digest] = label
                deduped.append(message)
                continue

            label = message.name or "tool"
            pointer = (
                f"[duplicate of {label} result {previous}: identical output omitted]"
            )
            if estimator(pointer) < estimator(message.content):
                deduped.append(_copy_message(message, content=pointer))
            else:
                deduped.append(message)
        return deduped


def _summary_line(message: Message, item_chars: int) -> str:
    """Render one digest line for a message, or ``""`` when it adds nothing."""
    role = message.role
    if role is Role.SYSTEM:
        return ""
    if role is Role.TOOL:
        return f"- {message.name or 'tool'} -> {_snippet(message.content, item_chars)}"
    if role is Role.ASSISTANT:
        if message.tool_calls:
            rendered = ", ".join(
                f"{call.name}("
                f"{json.dumps(call.arguments, default=str, separators=(',', ':'))})"
                for call in message.tool_calls
            )
            return f"- requested {_snippet(rendered, item_chars)}"
        if message.content:
            return f"- assistant: {_snippet(message.content, item_chars)}"
        return ""
    if role is Role.USER and message.content:
        return f"- user: {_snippet(message.content, item_chars)}"
    return ""


def extractive_summary(
    messages: Sequence[Message],
    *,
    max_items: int = 12,
    item_chars: int = 160,
) -> str:
    """Condense *messages* into a compact, deterministic digest.

    One line is emitted per meaningful turn (tool request, observation, assistant
    reply), identical consecutive lines are collapsed, and only the ``max_items``
    most recent lines are kept so a truncated digest still favours recent work.
    No model call is required, which keeps compression free, offline and
    reproducible; pass an LLM backed callable to :class:`PromptCompressor` when
    abstractive summaries are preferred.
    """
    lines: List[str] = []
    for message in messages:
        line = _summary_line(message, item_chars)
        if not line:
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)

    omitted = max(0, len(lines) - max_items)
    if omitted:
        lines = lines[omitted:]
    if not lines:
        return ""

    header = "Earlier steps condensed to save tokens:"
    if omitted:
        header += f" {omitted} older item(s) omitted."
    return f"{header}\n" + "\n".join(lines)


@dataclass
class MessageGroup:
    """Atomic conversation unit used by :class:`PromptCompressor`.

    See :func:`group_messages` for the grouping rules.  ``pinned`` marks a group
    that compression must keep, either because it carries the instructions
    (system messages, the goal) or because it is part of the recent window that
    the model is still reasoning over.
    """

    messages: List[Message] = field(default_factory=list)
    pinned: bool = False

    @property
    def role(self) -> Role:
        """Role of the first message in the group."""
        return self.messages[0].role

    @property
    def is_system(self) -> bool:
        """True for prompt/instruction messages, which are always pinned."""
        return bool(self.messages) and self.role is Role.SYSTEM

    @property
    def is_tool_exchange(self) -> bool:
        """True for an assistant tool request plus its observations."""
        return bool(self.messages) and (
            self.messages[0].role is Role.ASSISTANT
            and bool(self.messages[0].tool_calls)
        )

    def tokens(self, estimator: Callable[[str], int] = estimate_tokens) -> int:
        """Estimated tokens of the whole group."""
        return estimate_messages_tokens(self.messages, estimator=estimator)

    def flatten(self) -> List[Message]:
        """The group's messages, in order."""
        return list(self.messages)


def group_messages(messages: Sequence[Message]) -> List[MessageGroup]:
    """Split *messages* into atomic groups.

    An assistant message that requests tools is kept together with the ``tool``
    messages answering it, because OpenAI compatible APIs reject a transcript in
    which the two are separated.  Every other message forms its own group, so
    compression that drops a group can never leave an orphaned tool result.
    """
    groups: List[MessageGroup] = []
    exchange: Optional[MessageGroup] = None
    for message in messages:
        if message.role is Role.TOOL and exchange is not None:
            exchange.messages.append(message)
            continue

        group = MessageGroup([message])
        groups.append(group)
        if message.role is Role.ASSISTANT and message.tool_calls:
            exchange = group
        else:
            exchange = None
    return groups


#: Strategies applied by :class:`PromptCompressor` when none are supplied.
#: All three are stateless, so a single instance is safe to share between runs.
DEFAULT_STRATEGIES: Tuple[Compressor, ...] = (
    WhitespaceNormalizer(),
    ToolOutputCompressor(),
    ObservationDedupe(),
)

#: Type of a digest factory: full message list plus the active config.
Summarizer = Callable[[Sequence[Message], CompressionConfig], str]


class PromptCompressor:
    """Reduce the token cost of an agent prompt without changing its intent.

    The pipeline is applied on every call and each stage is independently
    useful:

    1. run the message level :class:`Compressor` strategies (whitespace, tool
       output, duplicate observations),
    2. replace the un-pinned middle of the transcript with a single digest
       message produced by :func:`extractive_summary`,
    3. if a token budget is configured, shrink tool observations harder and then
       drop the oldest un-pinned groups until the prompt fits.

    System messages, the goal and the ``keep_recent`` most recent groups are
    pinned: they are never summarised away, and no strategy can split an
    assistant tool request from the observations answering it.

    Args:
        config: Tunables; defaults to :class:`CompressionConfig`.
        strategies: Override the default :data:`DEFAULT_STRATEGIES` sequence.
        summarizer: ``(messages, config) -> str`` digest factory, e.g. an LLM
            backed summariser.  Defaults to the free
            :func:`extractive_summary` wrapper.
        estimator: Token estimator; defaults to :func:`estimate_tokens`.  Pass a
            tokenizer based function for exact budgeting.
    """

    def __init__(
        self,
        *,
        config: Optional[CompressionConfig] = None,
        strategies: Optional[Sequence[Compressor]] = None,
        summarizer: Optional[Summarizer] = None,
        estimator: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self.config = config or CompressionConfig()
        self.strategies = list(
            DEFAULT_STRATEGIES if strategies is None else strategies
        )
        self._summarizer = summarizer
        self.estimator = estimator

    # -- public API ------------------------------------------------------
    def compress(self, messages: Sequence[Message]) -> CompressedPrompt:
        """Return the compressed prompt plus the savings it achieved."""
        original = list(messages)
        if not original:
            return CompressedPrompt([], CompressionStats())

        before_tokens = self.estimate(original)
        current = original
        applied: List[str] = []
        for strategy in self.strategies:
            candidate = list(
                strategy.compress(
                    current, config=self.config, estimator=self.estimator
                )
            )
            if self.estimate(candidate) < self.estimate(current):
                applied.append(strategy.name)
            current = candidate

        groups = self._pin_groups(group_messages(current))
        groups, summarized = self._summarize_older(groups, applied)
        groups = self._trim_to_budget(groups, applied)
        compressed = [message for group in groups for message in group.messages]

        stats = CompressionStats(
            before_tokens=before_tokens,
            after_tokens=self.estimate(compressed),
            before_messages=len(original),
            after_messages=len(compressed),
            dropped_messages=max(0, len(original) - len(compressed)),
            summarized_messages=summarized,
            strategies=applied,
        )
        if stats.saved_tokens < self.config.min_savings_tokens:
            # Compression did not pay off; hand back the caller's prompt as is.
            return CompressedPrompt(
                original,
                CompressionStats(
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    before_messages=len(original),
                    after_messages=len(original),
                ),
            )
        return CompressedPrompt(compressed, stats)

    def compress_messages(self, messages: Sequence[Message]) -> List[Message]:
        """Compressed messages only, for direct provider calls."""
        return self.compress(messages).messages

    def __call__(self, messages: Sequence[Message]) -> CompressedPrompt:
        """Alias for :meth:`compress`, so a compressor can act as a hook."""
        return self.compress(messages)

    def estimate(self, messages: Sequence[Message]) -> int:
        """Estimated tokens of *messages* under the configured estimator."""
        if len(messages) >= self.config.parallel_threshold:
            return estimate_messages_tokens_parallel(
                messages,
                estimator=self.estimator,
                max_workers=self.config.max_workers,
            )
        return estimate_messages_tokens(messages, estimator=self.estimator)

    def summarizer(
        self, messages: Sequence[Message], config: CompressionConfig
    ) -> str:
        """Produce the digest that replaces the un-pinned middle."""
        if self._summarizer is not None:
            return self._summarizer(messages, config)
        return extractive_summary(messages, max_items=config.summary_max_items)

    # -- internals -------------------------------------------------------
    def _flatten(self, groups: Sequence[MessageGroup]) -> List[Message]:
        """Concatenate the messages of *groups*, preserving order."""
        return [message for group in groups for message in group.messages]

    def _pin_groups(self, groups: List[MessageGroup]) -> List[MessageGroup]:
        """Mark system messages, the goal and the recent window as immovable."""
        recent_start = len(groups)
        if self.config.keep_recent > 0:
            recent_start = max(0, len(groups) - self.config.keep_recent)

        goal_index: Optional[int] = None
        if self.config.keep_goal:
            goal_index = next(
                (index for index, group in enumerate(groups) if not group.is_system),
                None,
            )

        for index, group in enumerate(groups):
            group.pinned = (
                group.is_system or index >= recent_start or index == goal_index
            )
        return groups

    def _summarize_older(
        self, groups: List[MessageGroup], applied: List[str]
    ) -> Tuple[List[MessageGroup], int]:
        """Swap the un-pinned middle for one digest message.

        Returns the new group list and how many messages were summarised.  The
        digest is inserted directly after the leading system groups so the
        assistant/tool sequencing of the retained window is left untouched.
        """
        if not self.config.summarize_older:
            return groups, 0

        dropped = self._flatten([g for g in groups if not g.pinned])
        if not dropped:
            return groups, 0

        digest = (self.summarizer(dropped, self.config) or "").strip()
        if not digest:
            return groups, 0

        remaining = [group for group in groups if group.pinned]
        insert_at = 0
        while insert_at < len(remaining) and remaining[insert_at].is_system:
            insert_at += 1
        remaining.insert(
            insert_at, MessageGroup([Message.system(digest)], pinned=False)
        )
        if "summary" not in applied:
            applied.append("summary")
        return remaining, len(dropped)

    def _trim_to_budget(
        self, groups: List[MessageGroup], applied: List[str]
    ) -> List[MessageGroup]:
        """Fit *groups* into the configured prompt budget.

        Tool observations are shrunk to ``tool_output_hard_limit`` before any
        message is deleted, because dropping context is the most expensive way
        to save tokens.  Only then are the oldest un-pinned groups removed; when
        even the pinned prefix alone exceeds the budget the prompt is returned
        unchanged, since deleting the goal or the recent window would break the
        run.
        """
        budget = self.config.prompt_budget
        if budget is None:
            return groups
        if self.estimate(self._flatten(groups)) <= budget:
            return groups

        shrunk = self._shrink_hard(groups)
        if self.estimate(self._flatten(shrunk)) < self.estimate(self._flatten(groups)):
            if "tool-output-hard" not in applied:
                applied.append("tool-output-hard")
        groups = shrunk

        while self.estimate(self._flatten(groups)) > budget:
            index = next(
                (i for i, group in enumerate(groups) if not group.pinned), None
            )
            if index is None:
                break
            groups.pop(index)
            if "drop-oldest" not in applied:
                applied.append("drop-oldest")
        return groups

    def _shrink_hard(self, groups: List[MessageGroup]) -> List[MessageGroup]:
        """Apply the stricter observation ceiling to tool messages."""
        limit = self.config.tool_output_hard_limit
        if limit >= self.config.tool_output_limit:
            return groups

        shrunk: List[MessageGroup] = []
        for group in groups:
            messages: List[Message] = []
            for message in group.messages:
                if message.role is Role.TOOL and message.content:
                    content = compress_text(
                        message.content,
                        limit=limit,
                        json_max_items=self.config.json_max_items,
                        json_max_string=self.config.json_max_string,
                        estimator=self.estimator,
                    )
                    if content != message.content:
                        message = _copy_message(message, content=content)
                messages.append(message)
            shrunk.append(MessageGroup(messages, pinned=group.pinned))
        return shrunk

    def __repr__(self) -> str:
        names = ", ".join(strategy.name for strategy in self.strategies)
        return f"PromptCompressor(strategies=[{names}], config={self.config!r})"


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_STRATEGIES",
    "MESSAGE_OVERHEAD_TOKENS",
    "CompressedPrompt",
    "CompressionConfig",
    "CompressionStats",
    "Compressor",
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
]

