"""Tests for parallel processing features in the agent framework.

This module tests the parallel implementations added to compression,
tools, memory, and planner modules.
"""
from __future__ import annotations

import asyncio
import json
from typing import List

from .compression import (
    CompressionConfig,
    PromptCompressor,
    estimate_messages_tokens,
    estimate_messages_tokens_parallel,
    compress_text,
    compress_text_batch,
)
from .tools import Tool, ToolRegistry
from .memory import ConversationMemory
from .planner import TaskPlanner, extract_json_block, extract_json_blocks_parallel
from .types import Message, Role
from .benchmark import benchmark, benchmark_parallel_vs_sequential, BenchmarkSuite


def test_parallel_token_estimation():
    """Test parallel token estimation produces same results as sequential."""
    print("Testing parallel token estimation...")

    # Create test messages
    messages = [
        Message.user("This is a test message with some content." * 10)
        for _ in range(20)
    ]

    # Sequential estimation
    sequential_tokens = estimate_messages_tokens(messages)

    # Parallel estimation
    parallel_tokens = estimate_messages_tokens_parallel(messages)

    assert (
        sequential_tokens == parallel_tokens
    ), f"Token counts differ: {sequential_tokens} vs {parallel_tokens}"
    print(f"[OK] Sequential: {sequential_tokens} tokens, Parallel: {parallel_tokens} tokens")


def test_parallel_text_compression():
    """Test parallel text compression produces same results as sequential."""
    print("\nTesting parallel text compression...")

    # Create test texts
    texts = [
        json.dumps({"data": list(range(100)), "text": "x" * 500}) for _ in range(10)
    ]

    # Sequential compression
    sequential_results = [
        compress_text(text, limit=100, json_max_items=10, json_max_string=50)
        for text in texts
    ]

    # Parallel compression
    parallel_results = compress_text_batch(
        texts, limit=100, json_max_items=10, json_max_string=50
    )

    assert len(sequential_results) == len(
        parallel_results
    ), "Result count mismatch"
    for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
        assert seq == par, f"Result {i} differs: {seq} vs {par}"
    print(f"[OK] Compressed {len(texts)} texts successfully")


def test_parallel_tool_execution():
    """Test parallel tool execution."""
    print("\nTesting parallel tool execution...")

    # Create a simple tool
    call_count = 0

    def simple_tool(value: int) -> int:
        nonlocal call_count
        call_count += 1
        return value * 2

    tool = Tool(name="multiply", description="Multiply by 2", func=simple_tool)
    registry = ToolRegistry([tool])

    async def run_test():
        from .types import ToolCall

        calls = [
            ToolCall(id=f"call_{i}", name="multiply", arguments={"value": i})
            for i in range(5)
        ]

        # Sequential execution
        call_count = 0
        sequential_results = [
            await registry.run_call(call) for call in calls
        ]

        # Parallel execution
        call_count = 0
        parallel_results = await registry.run_calls_parallel(calls)

        assert len(sequential_results) == len(
            parallel_results
        ), "Result count mismatch"
        for seq, par in zip(sequential_results, parallel_results):
            assert seq.ok == par.ok, f"Success status differs: {seq.ok} vs {par.ok}"
        print(f"[OK] Executed {len(calls)} tool calls successfully")

    asyncio.run(run_test())


def test_parallel_memory_operations():
    """Test parallel memory operations."""
    print("\nTesting parallel memory operations...")

    # Create test messages
    messages = [
        Message.user(f"Test message {i} with some content." * 5) for i in range(15)
    ]

    # Sequential extend
    memory_seq = ConversationMemory(max_messages=20)
    memory_seq.extend(messages)

    # Parallel extend
    memory_par = ConversationMemory(max_messages=20)
    memory_par.extend_parallel(messages)

    assert len(memory_seq.messages()) == len(
        memory_par.messages()
    ), "Message count mismatch"
    print(f"[OK] Extended memory with {len(messages)} messages successfully")

    # Test parallel filtering
    def predicate(msg: Message) -> bool:
        return "Test message" in msg.content

    # Sequential filter
    seq_filtered = [msg for msg in memory_seq.messages() if predicate(msg)]

    # Parallel filter
    par_filtered = memory_seq.filter_parallel(predicate)

    assert len(seq_filtered) == len(
        par_filtered
    ), "Filter result count mismatch"
    print(f"[OK] Filtered {len(seq_filtered)} messages successfully")


def test_parallel_json_parsing():
    """Test parallel JSON parsing."""
    print("\nTesting parallel JSON parsing...")

    # Create text with multiple JSON blocks
    json_blocks = [
        json.dumps({"step": i, "description": f"Step {i}"}) for i in range(10)
    ]
    text = "\n".join(json_blocks)

    # Sequential parsing
    sequential_data = extract_json_block(text)

    # Parallel parsing
    parallel_data_list = extract_json_blocks_parallel(text)

    assert isinstance(parallel_data_list, list), "Parallel parsing should return list"
    assert len(parallel_data_list) > 0, "Should have parsed at least one JSON block"
    print(f"[OK] Parsed {len(parallel_data_list)} JSON blocks successfully")


def test_compression_config():
    """Test compression config with parallel settings."""
    print("\nTesting compression config with parallel settings...")

    config = CompressionConfig(
        parallel_threshold=5,
        max_workers=4,
    )

    assert config.parallel_threshold == 5
    assert config.max_workers == 4
    print("[OK] Compression config with parallel settings works")


def test_benchmark_suite():
    """Test the benchmark suite."""
    print("\nTesting benchmark suite...")

    suite = BenchmarkSuite("Parallel Features")

    def dummy_operation():
        sum(range(1000))

    # Run a benchmark
    result = suite.run(dummy_operation, "dummy", iterations=10)
    assert result.iterations == 10
    print(f"[OK] Benchmark suite: {result}")

    # Compare sequential vs parallel
    def sequential():
        [dummy_operation() for _ in range(10)]

    def parallel():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: dummy_operation(), range(10)))

    comparison = suite.compare(sequential, parallel, "dummy_comparison", iterations=5)
    print(f"[OK] Comparison: {comparison}")

    # Print summary
    print("\n" + suite.summary())


def run_all_tests():
    """Run all parallel feature tests."""
    print("=" * 60)
    print("Running Parallel Features Tests")
    print("=" * 60)

    try:
        test_parallel_token_estimation()
        test_parallel_text_compression()
        test_parallel_tool_execution()
        test_parallel_memory_operations()
        test_parallel_json_parsing()
        test_compression_config()
        test_benchmark_suite()

        print("\n" + "=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
