# LLM Agent Framework

A multi-step reasoning and autonomous task execution framework for AstroML.

## Overview

The Agent Framework enables AI agents to plan, reason, and execute complex tasks autonomously using multi-step reasoning chains, tool calling, and memory management.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Planner    │  │   Memory     │  │  Tool Registry│      │
│  │              │  │              │  │              │      │
│  │ - Decompose  │  │ - Context    │  │ - Functions  │      │
│  │ - Plan tasks │  │ - History   │  │ - Execution  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Multi-Step Reasoning Engine               │  │
│  │  - Think → Act → Observe → Reason Loop              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent

The main agent class that orchestrates planning, reasoning, and execution.

**Key Features:**
- Multi-step reasoning loop
- Tool/function calling
- Memory integration
- Planning integration
- Execution tracking

**Example:**
```python
from astroml.agents import Agent, AgentConfig
from astroml.agents.tools import DataLoader, DataAnalyzer

config = AgentConfig(
    name="my-agent",
    model="gpt-4",
    temperature=0.7,
    max_steps=10,
    enable_planning=True,
    enable_memory=True,
)

tools = [DataLoader(), DataAnalyzer()]
agent = Agent(config=config, tools=tools)

result = agent.run("Analyze the transaction data")
print(result.final_answer)
```

### 2. Memory Management

#### MemoryStore
Base memory implementation with message storage and retrieval.

```python
from astroml.agents.memory import MemoryStore

memory = MemoryStore()
memory.add_message("user", "Load the data")
memory.add_message("assistant", "Data loaded")

messages = memory.get_messages()
recent = memory.get_recent_messages(n=5)
```

#### ConversationMemory
Role-based conversation memory with summarization.

```python
from astroml.agents.memory import ConversationMemory

memory = ConversationMemory(max_history=100)
memory.add_message("user", "Task description")

user_msgs = memory.get_user_messages()
pairs = memory.get_dialogue_pairs()
summary = memory.summarize()
```

#### VectorMemory
Semantic memory with similarity search.

```python
from astroml.agents.memory import VectorMemory

memory = VectorMemory(embedding_dim=768)
memory.add_message("user", "Analyze fraud patterns")

# Semantic search
results = memory.similarity_search("detect anomalies", top_k=5)
context = memory.get_context_by_similarity("graph analysis")
```

### 3. Planning System

#### Planner
Base planner interface for task decomposition.

#### HeuristicPlanner
Rule-based planner using pattern matching.

```python
from astroml.agents.planning import HeuristicPlanner

planner = HeuristicPlanner()
plan = planner.plan("Train a fraud detection model")

for task in plan.tasks:
    print(f"{task.id}: {task.description}")
```

#### HierarchicalPlanner
Multi-level task decomposition.

```python
from astroml.agents.planning import HierarchicalPlanner

planner = HierarchicalPlanner(max_depth=3)
plan = planner.plan("Build and deploy fraud detection system")
```

### 4. Tool System

#### Tool Registry
Manages available tools for function calling.

```python
from astroml.agents.tools import ToolRegistry, DataLoader

registry = ToolRegistry()
registry.register(DataLoader())

tools = registry.list_tools()
tool = registry.get_tool("load_data")
```

#### Built-in Tools

**Data Tools:**
- `DataLoader`: Load data from files or databases
- `DataAnalyzer`: Analyze data and compute statistics
- `DataWriter`: Write data to destinations

**Graph Tools:**
- `GraphBuilder`: Build transaction graphs
- `GraphAnalyzer`: Analyze graph structure

**ML Tools:**
- `ModelTrainer`: Train ML models
- `ModelEvaluator`: Evaluate model performance

#### Custom Tools

Create custom tools by extending the `Tool` class:

```python
from astroml.agents.tools import Tool, ToolResult

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="A custom tool",
            parameters={
                "input": {
                    "type": "string",
                    "required": True,
                    "description": "Input parameter"
                }
            }
        )

    def execute(self, input_data, context=None):
        try:
            # Your tool logic here
            result = self.process(input_data)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def process(self, input_data):
        # Implement your logic
        return {"output": "processed result"}
```

Or use `FunctionTool` to wrap existing functions:

```python
from astroml.agents.tools import FunctionTool

def my_function(param1: str, param2: int) -> dict:
    return {"result": f"{param1}_{param2}"}

tool = FunctionTool(
    name="my_tool",
    description="My custom function",
    func=my_function,
    parameters={
        "param1": {"type": "string", "required": True},
        "param2": {"type": "number", "required": True},
    }
)
```

## CLI Usage

### Interactive Mode

```bash
python -m astroml agent interactive --agent-type fraud-detection
```

### Single Task Execution

```bash
python -m astroml agent run \
  --agent-type model-training \
  --task "Train a fraud detection model" \
  --context '{"data": "data.csv", "model_type": "sage"}' \
  --verbose
```

### List Available Tools

```bash
python -m astroml agent list-tools
```

## Pre-configured Agents

### Fraud Detection Agent

Specialized for fraud detection tasks on the Stellar network.

```python
from astroml.agents.examples.fraud_detection_agent import create_fraud_detection_agent

agent = create_fraud_detection_agent()
result = agent.run("Detect fraud patterns in recent transactions")
```

### Model Training Agent

Specialized for training ML models.

```python
from astroml.agents.examples.model_training_agent import create_model_training_agent

agent = create_model_training_agent()
result = agent.run("Train a GraphSAGE model for node classification")
```

## Execution Flow

1. **Task Input**: Agent receives a task description
2. **Planning**: Planner decomposes task into subtasks (if enabled)
3. **Memory Load**: Relevant context loaded from memory (if enabled)
4. **Reasoning Loop**:
   - Think: Generate reasoning about current state
   - Act: Decide on action or tool call
   - Observe: Execute tool and observe result
   - Update: Update memory and context
5. **Completion**: Return final answer when done or max steps reached

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "astroml-agent" | Agent name |
| `model` | str | "gpt-4" | LLM model to use |
| `temperature` | float | 0.7 | Sampling temperature |
| `max_steps` | int | 10 | Maximum reasoning steps |
| `max_tokens` | int | 2000 | Max tokens per response |
| `enable_planning` | bool | True | Enable task planning |
| `enable_memory` | bool | True | Enable memory |
| `enable_reflection` | bool | True | Enable reflection |
| `verbose` | bool | False | Verbose output |

## Integration with AstroML

The Agent Framework integrates seamlessly with existing AstroML components:

### Graph Building
```python
from astroml.agents.tools import GraphBuilder

# Uses astroml.features.transaction_graph internally
tool = GraphBuilder()
result = tool.execute({
    "transactions": "ledger_data",
    "window_size": 30,
    "node_type": "accounts"
})
```

### Model Training
```python
from astroml.agents.tools import ModelTrainer

# Uses astroml.training modules internally
tool = ModelTrainer()
result = tool.execute({
    "data": "graph_features",
    "model_type": "sage",
    "hyperparameters": {"hidden_dim": 128}
})
```

### Feature Engineering
```python
from astroml.agents.tools import DataAnalyzer

# Uses astroml.features modules internally
tool = DataAnalyzer()
result = tool.execute({
    "data": "transaction_data",
    "analysis_type": "correlation"
})
```

## Testing

Run the test suite:

```bash
pytest astroml/agents/__tests__/test_agent.py -v
```

## Examples

See the `astroml/agents/examples/` directory for complete examples:
- `fraud_detection_agent.py`: Fraud detection workflow
- `model_training_agent.py`: Model training workflow

## Advanced Usage

### Custom LLM Client

Provide your own LLM client:

```python
class CustomLLMClient:
    def generate(self, prompt, model, temperature, max_tokens):
        # Your implementation
        return response

agent = Agent(
    config=config,
    tools=tools,
    llm_client=CustomLLMClient()
)
```

### Multi-Agent Coordination

Coordinate multiple agents:

```python
agent1 = Agent(config=config1, tools=tools1)
agent2 = Agent(config=config2, tools=tools2)

result1 = agent1.run("Load and preprocess data")
result2 = agent2.run(
    "Train model",
    context={"data": result1.final_answer}
)
```

### State Persistence

Save and restore agent state:

```python
import json

# Save execution trace
trace = agent.get_execution_trace()
with open("trace.json", "w") as f:
    json.dump(trace, f)

# Save memory
memory_data = memory.get_messages()
with open("memory.json", "w") as f:
    json.dump([msg.to_dict() for msg in memory_data], f)
```

## Best Practices

1. **Start Simple**: Begin with simple tasks and increase complexity
2. **Use Planning**: Enable planning for complex multi-step tasks
3. **Monitor Memory**: Set appropriate memory limits for long conversations
4. **Tool Design**: Keep tools focused and well-documented
5. **Error Handling**: Implement robust error handling in custom tools
6. **Validation**: Use parameter validation in tools
7. **Testing**: Test tools independently before using in agents

## Limitations

- LLM integration requires API keys (not included in framework)
- Mock LLM mode for testing without actual LLM
- Vector memory uses simple hash-based embeddings (production should use real embeddings)
- Tool execution is synchronous (async support planned)

## Future Enhancements

- Async tool execution
- Real embedding models for vector memory
- Multi-agent communication protocols
- Reinforcement learning for action selection
- Tool composition and chaining
- Streaming responses
- Tool permission system
- Execution timeouts and resource limits

## License

MIT License - See main project LICENSE file.
