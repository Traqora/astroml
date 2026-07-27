"""LLM Providers — text generation and embeddings."""
from .factory import get_llm_provider
from .embedding_base import EmbeddingProvider, EmbeddingError
from .embedding_openai import OpenAIEmbeddingProvider
from .embedding_cohere import CohereEmbeddingProvider
from .embedding_huggingface import HuggingFaceEmbeddingProvider
from .embedding_local import LocalEmbeddingProvider
from .embedding_router import EmbeddingRouter, build_default_router

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
