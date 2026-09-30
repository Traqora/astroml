"""Abstract base classes for AstroML services (issue #573).

Defines the contracts for key services to enable:
- Dependency injection
- Implementation swapping
- Mock implementations for testing
- Clear interface documentation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd


@dataclass
class IngestionResult:
    """Result of an ingestion operation.

    Attributes:
        attempted: List of items that were attempted
        processed: List of items successfully processed
        skipped: List of items skipped (already processed)
        start_time: When ingestion started
        end_time: When ingestion completed
        errors: List of errors encountered
    """

    attempted: List[Any]
    processed: List[Any]
    skipped: List[Any]
    start_time: datetime
    end_time: datetime
    errors: List[str]

    @property
    def duration_seconds(self) -> float:
        """Duration of ingestion in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Percentage of attempted items that were processed."""
        if not self.attempted:
            return 0.0
        return len(self.processed) / len(self.attempted) * 100


@dataclass
class ComputationResult:
    """Result of a feature computation operation.

    Attributes:
        feature_name: Name of the computed feature
        data: Computed feature data as DataFrame
        metadata: Additional metadata about the computation
        computation_time_ms: Time taken to compute in milliseconds
        success: Whether computation succeeded
        error: Error message if computation failed
    """

    feature_name: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    computation_time_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class Graph:
    """Abstract graph representation.

    Attributes:
        nodes: List of node identifiers
        edges: List of edge tuples (source, target, attributes)
        directed: Whether the graph is directed
        metadata: Additional graph metadata
    """

    nodes: List[Any]
    edges: List[tuple[Any, Any, Dict[str, Any]]]
    directed: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> Any:
        if self.metadata is None:
            self.metadata = {}


class Ingestor(ABC):
    """Abstract base class for data ingestion services.

    Implementations should ingest data from various sources (e.g., Stellar ledgers,
    external APIs, files) into the system with idempotency guarantees.

    Example:
        class StellarIngestor(Ingestor):
            def ingest(self, start, end) -> Any:
                # Implementation
                return IngestionResult(...)
    """

    @abstractmethod
    def ingest(
        self,
        start: Any,
        end: Any,
        **kwargs: Any,
    ) -> IngestionResult:
        """Ingest data from start to end.

        Args:
            start: Starting point for ingestion (e.g., ledger ID, timestamp)
            end: Ending point for ingestion (e.g., ledger ID, timestamp)
            **kwargs: Additional implementation-specific parameters

        Returns:
            IngestionResult with summary of ingestion operation

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the ingestor.

        Returns:
            Dictionary with status information (e.g., last_processed, is_running)
        """
        pass

    def validate_config(self) -> bool:
        """Validate the ingestor configuration.

        Returns:
            True if configuration is valid

        Note:
            Default implementation returns True. Override for custom validation.
        """
        return True


class FeatureComputer(ABC):
    """Abstract base class for feature computation services.

    Implementations should compute features from raw data or other features.

    Example:
        class TransactionFeatureComputer(FeatureComputer):
            def compute(self, data) -> Any:
                # Implementation
                return pd.DataFrame(...)
    """

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute features from input data.

        Args:
            data: Input data (e.g., DataFrame, dict, custom object)
            **kwargs: Additional implementation-specific parameters

        Returns:
            DataFrame with computed features

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def get_feature_schema(self) -> Dict[str, Any]:
        """Get the schema of features produced by this computer.

        Returns:
            Dictionary mapping feature names to their types/descriptions
        """
        pass

    def validate_input(self, data: Any) -> bool:
        """Validate input data before computation.

        Args:
            data: Input data to validate

        Returns:
            True if data is valid for computation

        Note:
            Default implementation returns True. Override for custom validation.
        """
        return True


class GraphBuilder(ABC):
    """Abstract base class for graph construction services.

    Implementations should build graphs from transaction data or other sources.

    Example:
        class TransactionGraphBuilder(GraphBuilder):
            def build_graph(self, transactions) -> Any:
                # Implementation
                return Graph(...)
    """

    @abstractmethod
    def build_graph(self, transactions: Any, **kwargs: Any) -> Graph:
        """Build a graph from transaction data.

        Args:
            transactions: Transaction data (e.g., DataFrame, list of dicts)
            **kwargs: Additional implementation-specific parameters

        Returns:
            Graph object with nodes and edges

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def get_graph_statistics(self, graph: Graph) -> Dict[str, Any]:
        """Get statistics about a built graph.

        Args:
            graph: Graph to analyze

        Returns:
            Dictionary with graph statistics (e.g., node_count, edge_count)
        """
        pass


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for services that support caching.

    Implementations should provide cache management capabilities.
    """

    def cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key for the given arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            String cache key
        """
        ...

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate cache entries.

        Args:
            key: Specific cache key to invalidate, or None to clear all
        """
        ...


@runtime_checkable
class Observable(Protocol):
    """Protocol for services that support observability/metrics.

    Implementations should provide metrics collection capabilities.
    """

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary with metric names and values
        """
        ...

    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        ...
