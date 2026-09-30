# Parallel Features Baseline

This document describes the parallel processing features added to the AstroML agent framework to establish a performance baseline for concurrent operations.

## Overview

The parallel features baseline introduces concurrent processing capabilities across key agent components to improve performance for CPU-bound and IO-bound operations. All implementations include configurable thresholds to avoid overhead for small workloads.

## Components Enhanced

### 1. Compression Module (`compression.py`)

**Parallel Token Estimation**
- `estimate_messages_tokens_parallel()`: Parallelizes token estimation across messages using ThreadPoolExecutor
- Threshold: 10+ messages (below this, sequential processing is used)
- Configurable via `CompressionConfig.parallel_threshold` and `max_workers`

**Batch Text Compression**
- `compress_text_batch()`: Compresses multiple tool observations in parallel
- Threshold: 5+ texts
- Uses ThreadPoolExecutor for concurrent JSON parsing and text truncation

**Configuration Enhancements**
- `CompressionConfig.parallel_threshold`: Minimum messages for parallel processing (default: 10)
- `CompressionConfig.max_workers`: Maximum worker threads (default: CPU count)
- Automatic threshold detection in `PromptCompressor.estimate()`

### 2. Tools Module (`tools.py`)

**Parallel Tool Execution**
- `ToolRegistry.run_calls_parallel()`: Executes multiple tool calls concurrently
- Uses `asyncio.gather()` for optimal async tool performance
- Threshold: 3+ tool calls
- Configurable worker count

**Executor Integration**
- `AgentExecutor` automatically uses parallel execution when:
  - `AgentConfig.parallel_tool_execution` is enabled (default: True)
  - Multiple tool calls are made in a single step
  - Call count exceeds `AgentConfig.parallel_tool_threshold` (default: 2)

### 3. Memory Module (`memory.py`)

**Parallel Message Extension**
- `ConversationMemory.extend_parallel()`: Bulk message insertion with parallel validation
- Threshold: 10+ messages
- Parallelizes type checking and role validation
- Maintains order via index-based sorting

**Parallel Filtering**
- `ConversationMemory.filter_parallel()`: Filters messages using predicate evaluation in parallel
- Threshold: 20+ messages
- Useful for complex filtering operations on large conversation histories

### 4. Planner Module (`planner.py`)

**Parallel JSON Parsing**
- `extract_json_blocks_parallel()`: Extracts and parses multiple JSON blocks concurrently
- Threshold: 5+ JSON candidates
- Maintains original order via index-based sorting

**Parallel Plan Parsing**
- `TaskPlanner.parse_parallel()`: Parses plan steps in parallel for complex plans
- Threshold: 5+ steps
- Parallelizes step description parsing and argument extraction

### 5. Benchmark Module (`benchmark.py`)

**Performance Measurement**
- `BenchmarkResult`: Stores timing metrics for single operations
- `ComparisonResult`: Compares two benchmarks with speedup calculations
- `benchmark()`: Times function execution across multiple iterations
- `benchmark_parallel_vs_sequential()`: Direct comparison utility
- `BenchmarkSuite`: Manages collections of related benchmarks
- `timer()`: Context manager for quick timing

**Metrics Provided**
- Duration (total and average)
- Operations per second
- Speedup factor
- Percentage improvement
- Custom metadata support

## Configuration

### AgentConfig Enhancements
```python
@dataclass
class AgentConfig:
    # ... existing config ...
    parallel_tool_execution: bool = True
    parallel_tool_threshold: int = 2
    max_parallel_workers: Optional[int] = None
```

### CompressionConfig Enhancements
```python
@dataclass
class CompressionConfig:
    # ... existing config ...
    parallel_threshold: int = 10
    max_workers: Optional[int] = None
```

## Usage Examples

### Parallel Token Estimation
```python
from astroml.agent import estimate_messages_tokens_parallel

tokens = estimate_messages_tokens_parallel(
    messages,
    max_workers=4,
)
```

### Parallel Tool Execution
```python
from astroml.agent import AgentConfig, AgentExecutor

config = AgentConfig(
    parallel_tool_execution=True,
    parallel_tool_threshold=2,
    max_parallel_workers=4,
)

executor = AgentExecutor(llm, tools, config=config)
result = executor.run(goal)  # Automatically uses parallel when applicable
```

### Parallel Memory Operations
```python
from astroml.agent import ConversationMemory

memory = ConversationMemory(max_messages=100)
memory.extend_parallel(messages, max_workers=4)

filtered = memory.filter_parallel(
    lambda msg: "important" in msg.content,
    max_workers=4,
)
```

### Benchmarking
```python
from astroml.agent import benchmark, benchmark_parallel_vs_sequential

# Single benchmark
result = benchmark(my_function, "my_operation", iterations=10)
print(result)

# Comparison
comparison = benchmark_parallel_vs_sequential(
    sequential_func,
    parallel_func,
    "my_comparison",
    iterations=5,
)
print(comparison)
```

## Testing

Comprehensive tests are available in `test_parallel.py`:

```bash
python -m astroml.agent.test_parallel
```

Tests cover:
- Parallel token estimation correctness
- Parallel text compression
- Parallel tool execution
- Parallel memory operations
- Parallel JSON parsing
- Configuration validation
- Benchmark suite functionality

## Performance Considerations

### Thresholds
Each parallel operation includes a threshold to avoid overhead for small workloads:
- Token estimation: 10+ messages
- Text compression: 5+ texts
- Tool execution: 3+ calls
- Memory operations: 10-20+ messages
- JSON parsing: 5+ candidates

### Worker Count
Default worker count is CPU count (via `None`). This can be configured per operation:
- Too few workers: Underutilizes CPU
- Too many workers: Context switching overhead
- For IO-bound operations: Can exceed CPU count
- For CPU-bound operations: Match CPU count

### When to Use Parallel
**Good candidates:**
- Large message batches (> threshold)
- IO-bound operations (network, file I/O)
- Independent operations (no dependencies)
- CPU-intensive parsing/validation

**Poor candidates:**
- Small workloads (< threshold)
- Operations with dependencies
- Simple arithmetic operations
- Already-optimized sequential code

## Baseline Metrics

The benchmark suite provides baseline metrics for comparison. Run tests to establish your system's baseline:

```bash
python -m astroml.agent.test_parallel
```

Expected results will vary by hardware, but parallel operations should show:
- Token estimation: 2-4x speedup for 20+ messages
- Text compression: 1.5-3x speedup for 10+ texts
- Tool execution: Near-linear speedup for IO-bound tools
- Memory operations: 1.5-2x speedup for large batches

## Future Enhancements

Potential areas for further parallelization:
- LLM request batching (when provider supports it)
- Parallel plan execution (independent steps)
- Distributed tool execution across workers
- Async-native compression strategies
- GPU-accelerated token estimation

## Migration Guide

### Existing Code
No changes required - all parallel features are opt-in via configuration or explicit function calls.

### Enabling Parallel Features
1. Update `AgentConfig` to enable parallel tool execution
2. Update `CompressionConfig` to adjust parallel thresholds
3. Use parallel functions explicitly for custom operations
4. Benchmark before/after to validate improvements

### Disabling Parallel Features
Set `parallel_tool_execution=False` in `AgentConfig` or use threshold=0 in compression config to disable parallel processing.

## References

- Modified files:
  - `astroml/agent/compression.py`
  - `astroml/agent/tools.py`
  - `astroml/agent/memory.py`
  - `astroml/agent/planner.py`
  - `astroml/agent/executor.py`
  - `astroml/agent/types.py`
  - `astroml/agent/benchmark.py` (new)
  - `astroml/agent/test_parallel.py` (new)
