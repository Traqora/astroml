# LLM Agent Framework

AstroML ships an **LLM agent framework** for multi-step reasoning and
autonomous task execution on top of the existing graph pipeline. It is
provider agnostic, dependency light (standard library for the core loop),
and traces every step so autonomous runs stay auditable.

```
                      ┌──────────────────────────────┐
   goal ───────────▶  │  AgentExecutor (reason loop)  │
                      └───────────────┬──────────────┘
                                      │  messages
                          ┌───────────▼───────────┐
                          │   LLMProvider          │  EchoLLM
                          │   complete(messages)   │  ScriptedLLM
                          └───────────┬───────────┘  CallableLLM
                                      │ tool calls    OpenAICompatibleLLM
                          ┌───────────▼───────────┐
                          │   ToolRegistry         │  graph_overview
                          │   run(call) -> result  │  window_stats
                          └───────────┬───────────┘  account_features
                                      │ observations  top_accounts
                          ┌───────────▼───────────┐  score_accounts
                          │  Memory + AgentTrace   │
                          └───────────────────────┘
```

## Contents

- [Quick start](#quick-start)
- [Providers](#providers)
- [Tools](#tools)
- [Memory](#memory)
- [Planning](#planning)
- [The execution loop](#the-execution-loop)
- [AstroML domain tools](#astroml-domain-tools)
- [Command line interface](#command-line-interface)
- [Testing](#testing)

## Quick start

The agent core has no third-party requirements, so it can be exercised
offline with `ScriptedLLM` (deterministic replay) or `EchoLLM`.

```python
from astroml.agent import (
    AgentConfig,
    AgentExecutor,
    ToolRegistry,
    ScriptedLLM,
    tool,
)


@tool()
def count_edges(start_ts: int = 0, end_ts: int = 10) -> dict:
    """Count the transactions in an inclusive timestamp window."""
    return {"edges": 3}


llm = ScriptedLLM(
    [
        'Thought: count them\nAction: count_edges\nAction Input: {"start_ts": 1}',
        "Thought: done\nFinal Answer: 3 transactions",
    ]
)

agent = AgentExecutor(
    llm=llm,
    tools=ToolRegistry([count_edges]),
    config=AgentConfig(mode="react", max_steps=4),
)

result = agent.run("How many transactions are there?")
print(result.answer)          # 3 transactions
print(result.trace.summary())  # steps, tool calls, errors, duration
```

Every run returns an `AgentRunResult`:

| Field | Meaning |
|-------|---------|
| `result.answer` | Final answer, or `None` when the run failed |
| `result.success` | `True` only when `trace.status == "succeeded"` |
| `result.trace.steps` | Per-cycle thought, tool calls and observations |
| `result.trace.metadata` | Provider, model, mode, visible tools, plan |
| `result.to_dict()` | JSON-serialisable trace for logging/evaluation |

The loop **never raises** for model or tool failures — inspect
`trace.status` and `trace.error` instead.

## Providers

An agent only needs an object implementing `LLMProvider`:

```python
class LLMProvider(ABC):
    async def complete(self, messages, *, tools=None, temperature=0.0,
                       max_tokens=None) -> LLMResponse: ...
```

Four providers ship with the framework:

| Provider | Use case |
|----------|----------|
| `EchoLLM` | Offline default; echoes the latest user message |
| `ScriptedLLM` | Deterministic replay of a fixed response script |
| `CallableLLM` | Adapt any `messages -> response` function (sync or async) |
| `OpenAICompatibleLLM` | Any `/v1/chat/completions` endpoint via `aiohttp` |

`OpenAICompatibleLLM` works with OpenAI, Ollama, vLLM, LM Studio and
OpenRouter. `aiohttp` is imported lazily, so the rest of the framework keeps
working when it is absent.

Select a provider from the environment with `provider_from_env()`:

```bash
# Local Ollama
export ASTROML_LLM_PROVIDER=ollama
export ASTROML_LLM_MODEL=llama3.1
export ASTROML_LLM_BASE_URL=http://localhost:11434/v1

# OpenAI (or any compatible gateway)
export ASTROML_LLM_PROVIDER=openai
export ASTROML_LLM_MODEL=gpt-4o-mini
export ASTROML_LLM_API_KEY=sk-...
```

| Variable | Purpose |
|----------|---------|
| `ASTROML_LLM_PROVIDER` | `echo` (default), `scripted`, `openai`, `openai-compatible`, `ollama`, `vllm`, `lmstudio` |
| `ASTROML_LLM_MODEL` | Model identifier |
| `ASTROML_LLM_BASE_URL` | API root, e.g. `http://localhost:11434/v1` |
| `ASTROML_LLM_API_KEY` | Bearer token when the endpoint needs one |
| `ASTROML_LLM_TIMEOUT` | Request timeout in seconds (default 60) |
| `ASTROML_LLM_SCRIPT` | `scripted` provider script, `||` separated |
| `ASTROML_LLM_ON_EXHAUSTED` | `raise` \| `repeat_last` \| `empty` |

## Tools

A `Tool` wraps a plain Python callable plus the metadata an LLM needs.
Parameters schemas are derived from type hints; docstring first lines become
descriptions.

```python
from typing import Any, Dict, List, Optional

from astroml.agent import Tool, ToolError, ToolRegistry, tool


@tool(requires_confirmation=True)
def window_stats(
    edges: List[Dict[str, Any]],
    start_ts: int,
    end_ts: int,
    asset: Optional[str] = None,
) -> Dict[str, Any]:
    """Count nodes and edges in an inclusive timestamp window."""
    if start_ts > end_ts:
        raise ToolError("start_ts must be <= end_ts")
    ...
```

The generated JSON Schema for `window_stats` is:

```json
{
  "type": "object",
  "properties": {
    "edges":    {"type": "array", "items": {"type": "object"}},
    "start_ts": {"type": "integer"},
    "end_ts":   {"type": "integer"},
    "asset":    {"type": "string"}
  },
  "required": ["edges", "start_ts", "end_ts"]
}
```

Key behaviours:

- **Sync and async callables** are both supported; `Tool.run` awaits
  coroutine functions transparently.
- **Retryable failures.** Raise `ToolError("...", retryable=True)` to opt in
  to the retry budget (`AgentConfig.tool_retries`). Errors from tools are
  never fatal — they are rendered as `ERROR: ...` observations and fed back
  to the model so it can self-correct.
- **Unknown tools and bad arguments** become failed `ToolResult`s, not
  exceptions, because models hallucinate tool names.
- **Registry ordering is preserved** so prompts (and therefore runs) are
  deterministic. `registry.filtered(["window_stats"])` restricts the tool set
  exposed to the model.

## Memory

`ConversationMemory` is a bounded, system-message aware buffer. System
messages are pinned and always returned first; the remaining history is FIFO
capped at `max_messages`. `on_evict` is the hook for rolling
summarisation.

```python
from astroml.agent import ConversationMemory

memory = ConversationMemory(max_messages=40, on_evict=summarise_or_drop)
agent = AgentExecutor(llm, tools, memory=memory)
```

Passing a shared `memory` carries context across multiple runs; omitting it
gives each run a fresh buffer.

## Planning

`TaskPlanner` asks an LLM to decompose a goal into ordered steps using a
tolerant JSON contract (bare JSON, fenced blocks and JSON embedded in prose
all parse; unparseable output degrades to an empty plan rather than raising).

```python
from astroml.agent import AgentExecutor, TaskPlanner

planner = TaskPlanner(llm, max_steps=6)
agent = AgentExecutor(llm, tools, planner=planner)
result = agent.run("Find the busiest accounts and flag anything unusual")
```

When a plan is produced it is recorded on `result.trace.metadata["plan"]` and
injected into the prompt as a *suggested* plan — the executor still reasons
freely and may deviate.

## The execution loop

`AgentExecutor.arun` (async) and `AgentExecutor.run` (blocking wrapper)
implement reason → act → observe:

1. Send the goal, tool catalogue and (optionally) the plan to the model.
2. Interpret the turn. Three strategies are available via `AgentConfig.mode`:

   | Mode | Behaviour |
   |------|-----------|
   | `native` | Use provider function calling only |
   | `react` | Parse `Thought` / `Action` / `Action Input` / `Final Answer` text |
   | `auto` (default) | Prefer native tool calls, fall back to ReAct text |

3. Execute the requested tools, append observations, repeat.
4. Stop when the model answers directly, `max_steps` is reached, or the tool
   error budget is exhausted.

```python
from astroml.agent import AgentConfig

config = AgentConfig(
    max_steps=8,            # hard cap on reason/act cycles
    max_tool_errors=3,      # abort once more than N tool calls fail
    tool_retries=1,         # extra attempts for retryable ToolErrors
    stop_on_tool_error=False,
    temperature=0.0,
    max_tokens=None,
    mode="auto",
    include_tool_specs_in_prompt=True,
)
```

Statuses: `succeeded`, `failed`, `max_steps` (plus `pending`, `running`,
`skipped` for individual steps). Use `on_step=callback` for progress
reporting; callback exceptions are logged and never break the run.

## AstroML domain tools

`astroml.agent.domain_tools` wraps existing framework capabilities so an
agent can inspect graphs immediately:

| Tool | Purpose |
|------|---------|
| `graph_overview` | Node/edge counts, assets, density, busiest accounts |
| `window_stats` | Inclusive `[start_ts, end_ts]` slice via `window_snapshot` |
| `account_features` | Per-account features from `compute_node_features` |
| `top_accounts` | Rank accounts by volume, degree, in-degree or out-degree |
| `anomaly_score_tool` | Factory wrapping a trained `InductiveAnomalyScorer` |

```python
from astroml.agent import AgentExecutor, build_default_registry

tools = build_default_registry()                          # all four graph tools
tools = build_default_registry(include=["graph_overview"])  # subset
```

Graph tools accept edges as JSON mappings that follow the repository's edge
contract (`src`, `dst`, `timestamp`, optionally `amount` / `asset`).
Malformed input raises `ToolError`, which is surfaced to the model as an
observation.

Model-backed scoring stays opt-in so `torch` / `torch-geometric` are only
imported when a trained scorer exists:

```python
from astroml.agent.domain_tools import anomaly_score_tool

registry.register(anomaly_score_tool(my_trained_scorer))
```

## Command line interface

```bash
python -m astroml.agent "Summarise this transaction graph"
python -m astroml.agent --provider ollama --model llama3.1 --plan "Rank accounts"
python -m astroml.agent --edges data/edges.json --json "How many accounts?"
python -m astroml.cli agent --edges data/edges.json "..."   # same CLI
```

| Flag | Purpose |
|------|---------|
| `goal` / `--goal` | Goal for the agent to accomplish |
| `--edges PATH` | JSON edge list; binds the dataset so the model supplies only scalars |
| `--provider` `--model` `--base-url` `--api-key` | Provider overrides |
| `--mode {auto,native,react}` | Reasoning strategy |
| `--max-steps` `--max-tool-errors` | Loop bounds |
| `--plan` | Decompose the goal with `TaskPlanner` before executing |
| `--tools a,b` | Restrict the visible tool set |
| `--json` | Print the full trace as JSON |
| `--quiet` | Suppress per-step progress and the trace summary |

Exit codes: `0` success, `1` run failed, `2` configuration or IO error.
Progress is written to stderr; the answer goes to stdout.

## Testing

Agent tests are hermetic — they use `ScriptedLLM` and stub scorers and never
touch the network.

```bash
python -m pytest tests/test_agent_types.py tests/test_agent_tools.py \
  tests/test_agent_memory.py tests/test_agent_llm.py \
  tests/test_agent_planner.py tests/test_agent_executor.py \
  tests/test_agent_domain_tools.py tests/test_agent_cli.py -v
```

Feature-tool tests use `pytest.importorskip("pandas")`, so the suite passes
in environments without the ML stack installed.

## Design notes and limitations

- The core loop has no third-party dependencies. `aiohttp` is required only by
  `OpenAICompatibleLLM`; `pandas` / `torch` only by specific domain tools.
- Importing `astroml.agent.domain_tools` pulls in `astroml.features`, which
  eagerly imports the SQLAlchemy-backed pipelines. The domain tools are
  therefore re-exported lazily from `astroml.agent` via `__getattr__`.
- `requires_confirmation` marks destructive tools, but the framework does not
  enforce approval — host applications decide how to gate those calls.
- Tool output is truncated to 4000 characters before it reaches the model
  (`render_result`); summarise large payloads inside the tool instead.
- Tool execution is sequential by design: it keeps traces deterministic and
  avoids hammering endpoints from a single agent turn.


