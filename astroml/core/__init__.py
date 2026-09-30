"""Core abstractions and interfaces for AstroML.

This module provides abstract base classes and protocols that define
the contracts for key services in the system, enabling:
- Dependency injection
- Easy implementation swapping
- Mock implementations for testing
- Clear interface documentation
"""

from __future__ import annotations

from .abstracts import (
    ComputationResult,
    FeatureComputer,
    Graph,
    GraphBuilder,
    IngestionResult,
    Ingestor,
)

__all__ = [
    "Ingestor",
    "FeatureComputer",
    "GraphBuilder",
    "IngestionResult",
    "ComputationResult",
    "Graph",
]
