"""Example external plugin for the AstroML Feature Computer plugin system.

This demonstrates how to create a third-party feature computer plugin
that can be discovered and registered via the plugin architecture.

To use this plugin:
1. Install your package with this entry point in pyproject.toml:
   [project.entry-points."astroml.feature_computers"]
   my_plugin = "my_package.plugins"
2. The FeatureRegistry and ComputationEngine will auto-discover it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from astroml.features.feature_engine import (
    BaseFeatureComputer,
    FeatureDependencyType,
)
from astroml.cache import cached_feature


class CustomFeatureComputer(BaseFeatureComputer):
    """Example plugin computer that computes a custom metric.

    This computer can be registered via the plugin system by adding
    an entry point in your package's pyproject.toml:

    [project.entry-points."astroml.feature_computers"]
    custom = "examples.plugin_example"

    Once registered, the ComputationEngine will auto-discover and
    register this computer on initialization.
    """

    def __init__(self, window: int = 7):
        super().__init__("custom_metric")
        self.window = window
        self.add_dependency(
            "input_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "value", "timestamp"]},
        )

    @cached_feature(ttl_seconds=900)
    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute a rolling custom metric.

        Args:
            data: Input DataFrame with entity_id, value, timestamp columns
            entity_col: Entity identifier column name
            timestamp_col: Timestamp column name
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed custom metric indexed by entity
        """
        self.validate_input(data, entity_col, timestamp_col)

        value_col = kwargs.get("value_col", "value")
        result = (
            data.groupby(entity_col)[value_col]
            .rolling(window=self.window, min_periods=1)
            .mean()
        )
        result_df = pd.DataFrame({"custom_metric": result})
        result_df.index = data[entity_col].values
        return result_df


def create_custom_feature_computer(window: int = 7) -> CustomFeatureComputer:
    """Factory function for creating a configured plugin computer.

    This can also be registered as an entry point if it follows the
    FeatureComputer protocol (callable with data, entity_col, timestamp_col).

    Args:
        window: Rolling window size

    Returns:
        Configured CustomFeatureComputer instance
    """
    return CustomFeatureComputer(window=window)


class CallablePluginComputer:
    """Example of a callable-based plugin (non-class) for the plugin system.

    Plugins can also be simple callables that match the FeatureComputer protocol.
    """

    name = "callable_plugin_example"

    def __call__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute a simple feature.

        Args:
            data: Input DataFrame
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed feature
        """
        result = pd.DataFrame(
            {"callable_feature": data.groupby(entity_col).size()},
            index=data[entity_col].values,
        )
        return result


# For direct use in tests or scripts:
if __name__ == "__main__":
    import numpy as np

    # Create sample data
    sample_data = pd.DataFrame({
        "entity_id": ["a", "a", "b", "b", "a"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
    })

    # Test the plugin computer
    computer = create_custom_feature_computer(window=2)
    result = computer.compute(sample_data, entity_col="entity_id", timestamp_col="timestamp")
    print(f"Plugin result:\n{result}")

    # Test callable plugin
    callable_plugin = CallablePluginComputer()
    result2 = callable_plugin(sample_data, entity_col="entity_id", timestamp_col="timestamp")
    print(f"Callable plugin result:\n{result2}")
