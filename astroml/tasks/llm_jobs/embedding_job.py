"""Embedding backfill job — generate embeddings for historical data."""

from typing import Any
from astroml.llm.providers.base import LLMProvider


class EmbeddingJobHandler:
    """Generate embeddings for historical transactions."""

    type = "embedding"
    description = "Generate embeddings for all historical transactions"

    async def process_item(self, item: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
        text = item.get("text", "") or str(item.get("id", ""))
        vector = provider.embed(text)
        return {"embedding": vector, "dimensions": len(vector)}
