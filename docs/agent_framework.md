# LLM Agent Framework

A modular framework for multi-step reasoning and autonomous task execution, built into AstroML.

## Overview

The LLM Agent Framework provides a complete agent architecture with:

- **Multi-step reasoning** via ReAct, Chain-of-Thought, and Planner agent types
- **Autonomous task execution** with planning, recovery, and progress tracking
- **Three-tier memory** (short-term, long-term, episodic)
- **Pluggable LLM providers** (OpenAI, Anthropic, Mock)
- **Built-in tools** (calculator, Python REPL, file I/O, search, HTTP)
- **Configuration** via YAML, environment variables, or dataclasses
- **CLI integration** via `astroml agent` and `python -m astroml.agent`

## Quick Start

### Run a task

```bash
# Using the mock provider (no API key needed)
python -m astroml.agent run "Calculate the area of a circle with radius 5"

# Using OpenAI
python -m astroml.agent run "List files in the current directory" --provider openai --api-key YOUR_KEY

# Verbose output with JSON
python -m astroml.agent run "Explain dynamic graph learning" --verbose --json

# List available tools
python -m astroml.agent tools
```

### Programmatic usage

```python
from astroml.agent import AutonomousExecutor, create_agent, Task

# High-level: run a task autonomously
executor = AutonomousExecutor()
result = executor.run("Calculate the sum of 1 to 100")
print(result.output)

# Mid-level: use a specific agent type
agent = create_agent(agent_type="planner")
response = agent.run("Break down the concept of fraud detection")

# Low-level: use individual components
from astroml.agent import LLMClient, ToolRegistry, MemoryManager, CalculatorTool
from astroml.agent.config import LLMConfig, AgentConfig

llm = LLMClient(LLMConfig(provider="mock"))
registry = ToolRegistry()
registry.register(CalculatorTool())
memory = MemoryManager()

response = llm.chat([{"role": "user", "content": "What is 2 + 2?"}])
print(response.content)
```

## Architecture

```
astroml/agent/
├── __init__.py        # Package exports
├── __main__.py        # CLI entry point
├── base.py            # Agent base class + ReAct/CoT/Planner agents
├── config.py          # Configuration dataclasses
├── exceptions.py      # Custom exceptions
├── executor.py        # TaskExecutor + AutonomousExecutor
├── llm.py             # LLM provider abstraction (OpenAI, Anthropic, Mock)
├── memory.py          # MemoryManager + ShortTerm/LongTerm/Episodic memory
└── tools.py           # ToolRegistry + built-in tools
```

### Agent Types

| Type | Class | Description |
|------|-------|-------------|
| `react` | `ReActAgent` | ReAct pattern: reason → act → observe in a loop |
| `cot` | `ChainOfThoughtAgent` | Generates explicit step-by-step reasoning before acting |
| `planner` | `PlannerAgent` | Decomposes tasks into sub-steps, executes each, verifies |

### Memory Tiers

| Memory | Class | Description |
|--------|-------|-------------|
| Short-term | `ShortTermMemory` | Sliding-window buffer of recent messages (FIFO eviction) |
| Long-term | `LongTermMemory` | Key-value store with tagging and search |
| Episodic | `EpisodicMemory` | Log of completed task episodes for learning |

### Built-in Tools

| Tool | Name | Description |
|------|------|-------------|
| `CalculatorTool` | `calculator` | Safe arithmetic expression evaluation |
| `PythonREPLTool` | `python_repl` | Execute Python code in a subprocess |
| `FileReadTool` | `read_file` | Read file contents |
| `FileWriteTool` | `write_file` | Write content to a file |
| `ListDirectoryTool` | `list_directory` | List directory contents |
| `SearchTool` | `search` | Regex search in files |
| `HTTPRequestTool` | `http_request` | HTTP GET/POST requests |

## Configuration

### YAML config (`configs/agent.yaml`)

```yaml
agent_type: "planner"
llm:
  provider: "mock"  # or "openai", "anthropic"
  model: "gpt-3.5-turbo"
  api_key: null     # or set via env var
executor:
  max_steps: 50
  task_timeout: 600.0
  recovery_attempts: 2
```

### Environment variables

```bash
export ASTROML_AGENT_LLM_PROVIDER=openai
export ASTROML_AGENT_LLM_API_KEY=your-key-here
export ASTROML_AGENT_TYPE=planner
export ASTROML_AGENT_VERBOSE=true
```

### Programmatic

```python
from astroml.agent.config import AgentConfig, LLMConfig, ExecutorConfig

config = AgentConfig(
    llm=LLMConfig(provider="openai", api_key="your-key"),
    executor=ExecutorConfig(max_steps=100),
    agent_type="planner",
)
```

## CLI Reference

### `astroml agent run`

```
astroml agent run "Task description" [options]

Options:
  --agent-type [react|cot|planner]  Agent type (default: planner)
  --provider [mock|openai|anthropic]  LLM provider (default: mock)
  --model MODEL                     Model name
  --api-key KEY                     API key
  --max-steps N                     Max reasoning steps
  --verbose                         Verbose output
  --json                            JSON output
```

### `astroml agent tools`

Lists all available tools.

### `python -m astroml.agent`

Full CLI with `run`, `interactive`, `tools`, and `batch` subcommands.

## Testing

```bash
python -m pytest tests/test_agent.py -v
```

The test suite covers:
- Configuration loading and validation
- Memory subsystems (all three tiers)
- All built-in tools
- LLM mock provider
- All three agent types
- Task executor and autonomous executor
- Integration tests

## Custom Tools

```python
from astroml.agent import Tool, ToolResult, ToolRegistry

class MyCustomTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"}
        },
        "required": ["input"],
    }

    def execute(self, input: str) -> ToolResult:
        return ToolResult(success=True, output=f"Processed: {input}")

registry = ToolRegistry()
registry.register(MyCustomTool())
```

## License

MIT License — see the project root for details.
