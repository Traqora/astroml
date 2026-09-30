"""LLM Provider abstraction layer for AstroML.

This module provides a unified interface for LLM operations including:
- Multi-provider embedding generation (OpenAI, Cohere, local models)
- Embedding caching and drift monitoring
- LLM feature generation for fraud detection
- Tool execution and permission management
- Conversation memory and context management

Key components:
- providers: Embedding provider implementations and routing
- features: LLM-based feature computation
- fine_tuning: Model fine-tuning utilities
- testing: LLM testing utilities and fixtures
- tools: Tool execution framework with permission checks

Exports:
- EmbeddingProvider: Base embedding provider interface
- EmbeddingRouter: Multi-provider routing logic
- EmbeddingCache: Caching layer for embeddings
- ConversationMemory: Chat history management

Dependencies:
- openai: OpenAI API client
- cohere: Cohere API client
- sentence-transformers: Local embedding models
"""

from .blockchain_context import BlockchainContextBuilder
from .embedding_cache import EmbeddingCache, EmbeddingCacheStats
from .embedding_drift import (
    DriftAlert,
    DriftAlerter,
    DriftDetector,
    DriftReport,
    EmbeddingDistributionTracker,
    EmbeddingDriftMonitor,
)
from .memory import ConversationMemory
from .providers.embedding_base import EmbeddingError, EmbeddingProvider
from .providers.embedding_router import EmbeddingRouter, build_default_router
from .tools import (
    BaseTool,
    PermissionChecker,
    ToolAuditLog,
    ToolExecutor,
    ToolRegistry,
    get_global_registry,
)

__all__ = [
    "BlockchainContextBuilder",
    "ConversationMemory",
    "EmbeddingCache",
    "EmbeddingCacheStats",
    "EmbeddingDriftMonitor",
    "DriftDetector",
    "DriftReport",
    "DriftAlert",
    "DriftAlerter",
    "EmbeddingDistributionTracker",
    "EmbeddingProvider",
    "EmbeddingError",
    "EmbeddingRouter",
    "build_default_router",
    "features",
    "fine_tuning",
    "testing",
    "BaseTool",
    "ToolRegistry",
    "get_global_registry",
    "ToolExecutor",
    "PermissionChecker",
    "ToolAuditLog",
]
