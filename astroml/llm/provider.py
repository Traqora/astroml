import time
import asyncio
from typing import AsyncGenerator, Any
from abc import ABC, abstractmethod

class StreamingResponse:
    def __init__(self, async_generator: AsyncGenerator[str, None]):
        self.generator = async_generator
        
    async def get_chunks(self) -> AsyncGenerator[str, None]:
        async for chunk in self.generator:
            yield chunk

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
        
    @abstractmethod
    async def generate_stream(self, prompt: str) -> StreamingResponse:
        pass

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "This is a mock response."
        
    async def generate_stream(self, prompt: str) -> StreamingResponse:
        async def mock_generator() -> AsyncGenerator[str, None]:
            words = ["This", " is", " a", " simulated", " streaming", " response", "."]
            for word in words:
                await asyncio.sleep(0.05) # Simulate latency well within 200ms
                yield word
                
        return StreamingResponse(mock_generator())
