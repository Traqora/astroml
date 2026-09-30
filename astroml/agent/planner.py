"""Task decomposition and ReAct-style reasoning parsing.

Two independent pieces live here:

* :class:`TaskPlanner` asks an LLM to break a high level goal into an ordered
  list of steps, using a tolerant JSON contract that survives the loose
  formatting produced by smaller models.
* :class:`ReActParser` extracts the ``Thought`` / ``Action`` /
  ``Action Input`` / ``Final Answer`` sections emitted by models that do not
  support native tool calling, so the same executor loop can drive both.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._async import run_sync
from .llm import LLMProvider
from .types import Message, ToolCall

logger = logging.getLogger(__name__)


def _balanced_candidates(text: str) -> List[str]:
    """Return every balanced ``{...}`` / ``[...]`` slice found in *text*."""
    candidates: List[str] = []
    stack: List[str] = []
    start: Optional[int] = None
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            if not stack:
                start = index
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                continue
            if stack[-1] != pairs[char]:
                stack.clear()
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def extract_json_block(text: str) -> Optional[Any]:
    """Return the first decodable JSON value embedded in *text*.

    Handles bare JSON, fenced ```json blocks, and JSON surrounded by prose.
    Returns ``None`` when nothing decodable is present.
    """
    if not text:
        return None

    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.extend(_balanced_candidates(text))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_json_blocks_parallel(
    text: str,
    *,
    max_workers: Optional[int] = None,
    threshold: int = 5,
) -> List[Any]:
    """Return all decodable JSON values embedded in *text* using parallel parsing.

    Handles bare JSON, fenced ```json blocks, and JSON surrounded by prose.
    Returns an empty list when nothing decodable is present.

    Args:
        text: Text containing JSON blocks.
        max_workers: Maximum number of worker threads for parallel parsing.
        threshold: Minimum number of candidates to enable parallel processing.

    Returns:
        List of decoded JSON values in the order they appear in the text.
    """
    if not text:
        return []

    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.extend(_balanced_candidates(text))

    if len(candidates) < threshold:
        results = []
        for candidate in candidates:
            try:
                results.append(json.loads(candidate))
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    def try_parse(candidate: str) -> Optional[Any]:
        """Try to parse a single JSON candidate."""
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(try_parse, candidate): idx
            for idx, candidate in enumerate(candidates)
        }
        results_with_indices = []
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            if result is not None:
                results_with_indices.append((idx, result))

    # Sort by original index to maintain order
    results_with_indices.sort(key=lambda x: x[0])
    return [result for _, result in results_with_indices]


def _join(lines: Sequence[str]) -> str:
    """Join collected section lines, trimming blank edges."""
    return "\n".join(line.strip() for line in lines).strip()


def _coerce_scalar(value: str) -> Any:
    """Convert a textual token into bool / None / int / float / str."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except (TypeError, ValueError):
            continue
    return value


def _parse_action_input(raw: str) -> Dict[str, Any]:
    """Decode an ``Action Input`` payload into a kwargs dictionary."""
    text = (raw or "").strip()
    if not text:
        return {}

    parsed = extract_json_block(text)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    if isinstance(parsed, list):
        return {"input": parsed}

    pairs = re.findall(
        r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,\n]+)",
        text,
    )
    if pairs:
        return {
            key: _coerce_scalar(value.strip().strip("\"'"))
            for key, value in pairs
        }
    return {"input": text}


_MARKER_RE = re.compile(
    r"^\s*(Thought|Action\s*Input|Action|Final\s*Answer)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_OBSERVATION_RE = re.compile(r"^\s*Observation\s*:", re.IGNORECASE)


@dataclass
class ReActOutput:
    """Structured view over a single ReAct-style model turn."""

    thought: str = ""
    action: Optional[str] = None
    action_input: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None

    @property
    def has_action(self) -> bool:
        return bool(self.action)

    def to_tool_call(self, *, call_id: Optional[str] = None) -> ToolCall:
        """Convert the parsed action into a :class:`ToolCall`."""
        if not self.action:
            raise ValueError("ReActOutput has no action to convert")
        name = self.action.strip()
        return ToolCall(
            id=call_id or f"call_{name}", name=name, arguments=dict(self.action_input)
        )


class ReActParser:
    """Parse ``Thought / Action / Action Input / Final Answer`` text.

    The parser is deliberately forgiving: markers are matched
    case-insensitively, action inputs may be JSON, ``key=value`` pairs or a
    bare string, and hallucinated trailing ``Observation`` sections are
    ignored.
    """

    @classmethod
    def parse(cls, text: str) -> ReActOutput:
        """Parse *text* into a :class:`ReActOutput`."""
        if not text or not text.strip():
            return ReActOutput()

        sections: Dict[str, List[str]] = {}
        current: Optional[str] = None
        for raw_line in text.splitlines():
            if _OBSERVATION_RE.match(raw_line):
                current = None
                continue
            match = _MARKER_RE.match(raw_line)
            if match:
                key = re.sub(r"\s+", " ", match.group(1).lower())
                sections.setdefault(key, []).append(match.group(2))
                current = key
                continue
            if current is not None:
                sections[current].append(raw_line)

        if not sections:
            # Plain prose with no markers: treat the whole turn as reasoning.
            return ReActOutput(thought=text.strip())

        thought = _join(sections.get("thought", []))
        action = _join(sections.get("action", []))
        final = _join(sections.get("final answer", []))

        return ReActOutput(
            thought=thought,
            action=action or None,
            action_input=_parse_action_input(_join(sections.get("action input", []))),
            final_answer=final if "final answer" in sections else None,
        )


_PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant for AstroML, a dynamic graph machine "
    "learning framework for the Stellar blockchain. Decompose the user's goal "
    "into the smallest useful ordered list of steps, preferring available "
    "tools over speculation. Reply with JSON only, matching this schema:\n"
    '{"steps": [{"description": "...", "tool": "<tool name or null>", '
    '"arguments": {}}], "notes": "..."}'
)


@dataclass
class PlanStep:
    """One unit of work produced by :class:`TaskPlanner`."""

    description: str
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"description": self.description, "tool": self.tool}
        if self.arguments:
            payload["arguments"] = dict(self.arguments)
        return payload


@dataclass
class Plan:
    """Ordered decomposition of a goal."""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    notes: str = ""
    raw: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "notes": self.notes,
        }

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


class TaskPlanner:
    """LLM driven task decomposition with a tolerant JSON contract."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_steps: int = 8,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.max_steps = max_steps
        self.temperature = temperature
        self.system_prompt = system_prompt or _PLANNER_SYSTEM_PROMPT

    def build_messages(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tool_names: Optional[Sequence[str]] = None,
    ) -> List[Message]:
        """Build the single-turn prompt used for a planning pass."""
        lines = [f"Goal: {goal}"]
        if context:
            lines.append(f"\nContext:\n{context}")
        if tool_names:
            lines.append("\nAvailable tools: " + ", ".join(tool_names))
        lines.append(f"\nReturn at most {self.max_steps} steps as JSON.")
        return [Message.system(self.system_prompt), Message.user("\n".join(lines))]

    async def aplan(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tool_names: Optional[Sequence[str]] = None,
    ) -> Plan:
        """Ask the LLM for a plan and parse it into a :class:`Plan`."""
        messages = self.build_messages(goal, context=context, tool_names=tool_names)
        response = await self.llm.complete(
            messages, tools=None, temperature=self.temperature
        )
        return self.parse(response.content, goal=goal, max_steps=self.max_steps)

    def plan(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tool_names: Optional[Sequence[str]] = None,
    ) -> Plan:
        """Blocking wrapper around :meth:`aplan`."""
        return run_sync(self.aplan(goal, context=context, tool_names=tool_names))

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        goal: str = "",
        max_steps: Optional[int] = None,
    ) -> Plan:
        """Parse a plan out of model text, tolerating fenced or loose JSON."""
        data = extract_json_block(text or "")
        raw_steps: Sequence[Any] = []
        notes = ""
        if isinstance(data, Mapping):
            raw_steps = data.get("steps") or data.get("plan") or []
            notes = str(data.get("notes") or "")
        elif isinstance(data, list):
            raw_steps = data

        steps: List[PlanStep] = []
        for item in raw_steps:
            if isinstance(item, str):
                if item.strip():
                    steps.append(PlanStep(description=item.strip()))
                continue
            if not isinstance(item, Mapping):
                continue
            description = (
                item.get("description") or item.get("step") or item.get("task") or ""
            )
            tool = item.get("tool") or item.get("tool_name")
            arguments = item.get("arguments") or {}
            steps.append(
                PlanStep(
                    description=str(description).strip(),
                    tool=str(tool) if tool else None,
                    arguments=(
                        dict(arguments) if isinstance(arguments, Mapping) else {}
                    ),
                )
            )

        if max_steps:
            steps = steps[:max_steps]
        return Plan(goal=goal, steps=steps, notes=notes, raw=text)

    @classmethod
    def parse_parallel(
        cls,
        text: str,
        *,
        goal: str = "",
        max_steps: Optional[int] = None,
        max_workers: Optional[int] = None,
        threshold: int = 5,
    ) -> Plan:
        """Parse a plan out of model text with parallel step processing.

        Uses ThreadPoolExecutor to parallelize step parsing when there are many steps.
        Beneficial for complex plans with numerous steps.

        Args:
            text: Model text containing the plan.
            goal: The original goal being planned.
            max_steps: Maximum number of steps to include.
            max_workers: Maximum number of worker threads.
            threshold: Minimum number of steps to enable parallel processing.

        Returns:
            Parsed Plan object.
        """
        data = extract_json_block(text or "")
        raw_steps: Sequence[Any] = []
        notes = ""
        if isinstance(data, Mapping):
            raw_steps = data.get("steps") or data.get("plan") or []
            notes = str(data.get("notes") or "")
        elif isinstance(data, list):
            raw_steps = data

        if len(raw_steps) < threshold:
            return cls.parse(text, goal=goal, max_steps=max_steps)

        def parse_step(item: Any) -> Optional[PlanStep]:
            """Parse a single step item."""
            if isinstance(item, str):
                if item.strip():
                    return PlanStep(description=item.strip())
                return None
            if not isinstance(item, Mapping):
                return None
            description = (
                item.get("description") or item.get("step") or item.get("task") or ""
            )
            tool = item.get("tool") or item.get("tool_name")
            arguments = item.get("arguments") or {}
            return PlanStep(
                description=str(description).strip(),
                tool=str(tool) if tool else None,
                arguments=(
                    dict(arguments) if isinstance(arguments, Mapping) else {}
                ),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(parse_step, item): idx
                for idx, item in enumerate(raw_steps)
            }
            steps_with_indices = []
            for future in as_completed(futures):
                idx = futures[future]
                step = future.result()
                if step is not None:
                    steps_with_indices.append((idx, step))

        # Sort by original index to maintain order
        steps_with_indices.sort(key=lambda x: x[0])
        steps = [step for _, step in steps_with_indices]

        if max_steps:
            steps = steps[:max_steps]
        return Plan(goal=goal, steps=steps, notes=notes, raw=text)

