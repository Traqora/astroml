"""LLM package entrypoints for AstroML.

The package exposes the public LLM interfaces lazily to keep the namespace import-safe
when optional dependencies are not installed or when only a subset of the LLM stack
is needed in tests.
"""

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
    "BaseTool",
    "ToolRegistry",
    "get_global_registry",
    "ToolExecutor",
    "PermissionChecker",
    "ToolAuditLog",
]


def __getattr__(name: str):
    if name == "BlockchainContextBuilder":
        from .blockchain_context import BlockchainContextBuilder
        return BlockchainContextBuilder
    if name in {"EmbeddingCache", "EmbeddingCacheStats"}:
        from .embedding_cache import EmbeddingCache, EmbeddingCacheStats
        return {"EmbeddingCache": EmbeddingCache, "EmbeddingCacheStats": EmbeddingCacheStats}[name]
    if name in {"DriftAlert", "DriftAlerter", "DriftDetector", "DriftReport", "EmbeddingDistributionTracker", "EmbeddingDriftMonitor"}:
        from .embedding_drift import (
            DriftAlert,
            DriftAlerter,
            DriftDetector,
            DriftReport,
            EmbeddingDistributionTracker,
            EmbeddingDriftMonitor,
        )
        return {
            "DriftAlert": DriftAlert,
            "DriftAlerter": DriftAlerter,
            "DriftDetector": DriftDetector,
            "DriftReport": DriftReport,
            "EmbeddingDistributionTracker": EmbeddingDistributionTracker,
            "EmbeddingDriftMonitor": EmbeddingDriftMonitor,
        }[name]
    if name == "ConversationMemory":
        from .memory import ConversationMemory
        return ConversationMemory
    if name in {"EmbeddingProvider", "EmbeddingError"}:
        from .providers.embedding_base import EmbeddingError, EmbeddingProvider
        return {"EmbeddingProvider": EmbeddingProvider, "EmbeddingError": EmbeddingError}[name]
    if name in {"EmbeddingRouter", "build_default_router"}:
        from .providers.embedding_router import EmbeddingRouter, build_default_router
        return {"EmbeddingRouter": EmbeddingRouter, "build_default_router": build_default_router}[name]
    if name in {"BaseTool", "PermissionChecker", "ToolAuditLog", "ToolExecutor", "ToolRegistry", "get_global_registry"}:
        from .tools import (
            BaseTool,
            PermissionChecker,
            ToolAuditLog,
            ToolExecutor,
            ToolRegistry,
            get_global_registry,
        )
        return {
            "BaseTool": BaseTool,
            "PermissionChecker": PermissionChecker,
            "ToolAuditLog": ToolAuditLog,
            "ToolExecutor": ToolExecutor,
            "ToolRegistry": ToolRegistry,
            "get_global_registry": get_global_registry,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
