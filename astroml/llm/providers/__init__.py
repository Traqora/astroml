"""LLM Providers — text generation and embeddings."""

from .embedding_base import EmbeddingError, EmbeddingProvider
from .embedding_cohere import CohereEmbeddingProvider
from .embedding_huggingface import HuggingFaceEmbeddingProvider
from .embedding_local import LocalEmbeddingProvider
from .embedding_openai import OpenAIEmbeddingProvider
from .embedding_router import EmbeddingRouter, build_default_router
from .factory import get_llm_provider

__all__ = [
    "get_llm_provider",
    # Embedding abstraction
    "EmbeddingProvider",
    "EmbeddingError",
    # Concrete adapters
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "LocalEmbeddingProvider",
    # Router / factory
    "EmbeddingRouter",
    "build_default_router",
]
