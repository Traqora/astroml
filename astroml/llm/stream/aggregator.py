import asyncio
from typing import AsyncGenerator, List, Dict, Any

class MultiSourceAggregator:
    def __init__(self):
        pass

    async def aggregate_streams(self, streams: List[AsyncGenerator[Dict[str, Any], None]]) -> AsyncGenerator[Dict[str, Any], None]:
        # Reads from multiple generators and outputs them as a combined stream tagged by source
        queue = asyncio.Queue()
        active_generators = len(streams)

        async def worker(index: int, gen: AsyncGenerator[Dict[str, Any], None]):
            nonlocal active_generators
            try:
                async for item in gen:
                    await queue.put({"source_index": index, "data": item})
            finally:
                active_generators -= 1
                if active_generators == 0:
                    await queue.put(None) # Sentinel

        # Start tasks
        for idx, s in enumerate(streams):
            asyncio.create_task(worker(idx, s))

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
