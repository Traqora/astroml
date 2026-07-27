"""Tests for batch processor."""

from astroml.llm.batch import BatchProcessor, CheckpointManager
from astroml.llm.batch.strategies import FixedSizeStrategy


class MockProvider:
    """Minimal mock provider for testing BatchProcessor."""

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def generate(self, prompt, **kwargs):
        return f"response to: {prompt[:20]}"


async def mock_process_fn(item, provider):
    return {"processed": item.get("id", 0)}


async def failing_process_fn(item, provider):
    raise ValueError("Intentional failure")


class TestBatchProcessor:
    def setup_method(self):
        self.provider = MockProvider()
        self.checkpoint = CheckpointManager("test_job")

    def test_process_range_empty(self):
        proc = BatchProcessor(self.provider, self.checkpoint, FixedSizeStrategy(10))
        import asyncio
        results = asyncio.run(proc.process_range([], mock_process_fn))
        assert results == []

    def test_process_range_single_item(self):
        proc = BatchProcessor(self.provider, self.checkpoint, FixedSizeStrategy(10))
        items = [{"id": 1}]
        import asyncio
        results = asyncio.run(proc.process_range(items, mock_process_fn))
        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert results[0]["result"]["processed"] == 1

    def test_process_range_multi_batch(self):
        proc = BatchProcessor(self.provider, self.checkpoint, FixedSizeStrategy(2))
        items = [{"id": i} for i in range(5)]
        import asyncio
        results = asyncio.run(proc.process_range(items, mock_process_fn))
        assert len(results) == 5
        assert all(r["status"] == "completed" for r in results)

    def test_process_range_with_failures(self):
        proc = BatchProcessor(self.provider, self.checkpoint, FixedSizeStrategy(5))
        items = [{"id": i} for i in range(3)]
        import asyncio
        results = asyncio.run(proc.process_range(items, failing_process_fn))
        assert len(results) == 3
        assert all(r["status"] == "failed" for r in results)
        assert self.checkpoint.get_progress()["failed"] == 3
