"""Mock implementations for testing (issue #573).

Provides mock implementations of the abstract base classes for use in tests.
These implementations allow testing without requiring real data sources or
external dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .abstracts import FeatureComputer, Graph, GraphBuilder, IngestionResult, Ingestor


class MockIngestor(Ingestor):
    """Mock ingestor for testing.

    Simulates ingestion without requiring real data sources.
    """

    def __init__(self, fail_on -> Any: Optional[List[int]] = None):
        """Initialize mock ingestor.

        Args:
            fail_on: List of item IDs that should fail during ingestion
        """
        self.fail_on = fail_on or []
        self.ingestion_count = 0

    def ingest(
        self,
        start: Any,
        end: Any,
        **kwargs: Any,
    ) -> IngestionResult:
        """Mock ingestion that simulates processing.

        Args:
            start: Starting point
            end: Ending point
            **kwargs: Additional parameters

        Returns:
            IngestionResult with mock data
        """
        start_time = datetime.utcnow()
        attempted = list(range(start, end + 1))
        processed = []
        skipped = []
        errors = []

        for item_id in attempted:
            self.ingestion_count += 1
            if item_id in self.fail_on:
                errors.append(f"Failed to process item {item_id}")
            else:
                processed.append(item_id)

        end_time = datetime.utcnow()

        return IngestionResult(
            attempted=attempted,
            processed=processed,
            skipped=skipped,
            start_time=start_time,
            end_time=end_time,
            errors=errors,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get mock status.

        Returns:
            Dictionary with mock status information
        """
        return {
            "last_processed": self.ingestion_count,
            "is_running": False,
            "mock": True,
        }


class MockFeatureComputer(FeatureComputer):
    """Mock feature computer for testing.

    Simulates feature computation without real logic.
    """

    def __init__(self, feature_name -> Any: str = "mock_feature"):
        """Initialize mock feature computer.

        Args:
            feature_name: Name of the feature to compute
        """
        self.feature_name = feature_name
        self.compute_count = 0

    def compute(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        """Mock feature computation.

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            DataFrame with mock features
        """
        self.compute_count += 1

        if isinstance(data, pd.DataFrame):
            # Return a DataFrame with a mock feature column
            result = data.copy()
            result[self.feature_name] = 1.0
            return result
        else:
            # Return a simple mock DataFrame
            return pd.DataFrame(
                {
                    self.feature_name: [1.0] * 10,
                    "id": list(range(10)),
                }
            )

    def get_feature_schema(self) -> Dict[str, Any]:
        """Get mock feature schema.

        Returns:
            Dictionary with mock schema
        """
        return {
            self.feature_name: {
                "type": "float64",
                "description": "Mock feature for testing",
            }
        }

    def validate_input(self, data: Any) -> bool:
        """Validate input data.

        Args:
            data: Input data to validate

        Returns:
            True if data is not None
        """
        return data is not None


class MockGraphBuilder(GraphBuilder):
    """Mock graph builder for testing.

    Simulates graph construction without real logic.
    """

    def __init__(self, directed -> Any: bool = True):
        """Initialize mock graph builder.

        Args:
            directed: Whether to build directed graphs
        """
        self.directed = directed
        self.build_count = 0

    def build_graph(self, transactions: Any, **kwargs: Any) -> Graph:
        """Mock graph construction.

        Args:
            transactions: Transaction data
            **kwargs: Additional parameters

        Returns:
            Graph with mock data
        """
        self.build_count += 1

        # Create a simple mock graph
        nodes = ["node_0", "node_1", "node_2"]
        edges = [
            ("node_0", "node_1", {"weight": 1.0}),
            ("node_1", "node_2", {"weight": 1.0}),
        ]

        return Graph(
            nodes=nodes,
            edges=edges,
            directed=self.directed,
            metadata={"mock": True, "build_count": self.build_count},
        )

    def get_graph_statistics(self, graph: Graph) -> Dict[str, Any]:
        """Get mock  graph statistics.

        Args:
            graph: Graph to analyze

        Returns:
            Dictionary with mock statistics
        """
        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "directed": graph.directed,
            "mock": True,
        }
