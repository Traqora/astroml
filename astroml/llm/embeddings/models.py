"""Embedding model configurations."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    OPENAI_LARGE = "text-embedding-3-large"
    OPENAI_SMALL = "text-embedding-3-small"
    COHERE_V3 = "embed-english-v3.0"
    COHERE_LIGHT = "embed-english-light-v3.0"
    SENTENCE_TRANSFORMERS = "all-MiniLM-L6-v2"
    BGEEMBEDDINGS = "BAAI/bge-large-en-v1.5"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding service."""

    model: EmbeddingModel = EmbeddingModel.OPENAI_LARGE
    embedding_dim: int = 3072
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 32
    similarity_metric: str = "cosine"
    max_batch_tokens: int = 8191
    cache_embeddings: bool = True
    api_timeout: int = 30


MODEL_DIMENSIONS = {
    EmbeddingModel.OPENAI_LARGE: 3072,
    EmbeddingModel.OPENAI_SMALL: 1536,
    EmbeddingModel.COHERE_V3: 1024,
    EmbeddingModel.COHERE_LIGHT: 384,
    EmbeddingModel.SENTENCE_TRANSFORMERS: 384,
    EmbeddingModel.BGEEMBEDDINGS: 1024,
}
