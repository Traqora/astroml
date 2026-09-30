"""Visualizations for automated data profiling.

Generates distribution plots, correlation heatmaps, missing-value charts
and outlier boxplots as matplotlib figures, optionally saving them to
disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ProfilingVisualizer:
    """Create exploratory visualizations from tabular data."""

    def distribution_plot(
        self,
        series: pd.Series,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot the distribution of a numeric column.

        Args:
            series: Numeric column to visualize.
            title: Optional plot title.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series.dropna().astype(float), bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(str(series.name))
        ax.set_ylabel("Frequency")
        ax.set_title(title or f"Distribution of {series.name}")
        ax.grid(True, alpha=0.3)
        self._save(fig, save_path)
        return fig

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot a correlation heatmap for numeric columns.

        Args:
            df: DataFrame containing numeric columns.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib figure.
        """
        numeric = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        if numeric.shape[1] == 0:
            ax.text(0.5, 0.5, "No numeric columns to correlate", ha="center")
            ax.axis("off")
            self._save(fig, save_path)
            return fig
        corr = numeric.corr()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Correlation Heatmap")
        self._save(fig, save_path)
        return fig

    def missing_value_plot(
        self,
        df: pd.DataFrame,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot missing value counts per column.

        Args:
            df: DataFrame to inspect.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib figure.
        """
        missing = df.isna().sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        if missing.sum() == 0:
            ax.text(0.5, 0.5, "No missing values", ha="center")
            ax.axis("off")
            self._save(fig, save_path)
            return fig
        missing = missing[missing > 0]
        ax.bar(missing.index.astype(str), missing.values, color="salmon", edgecolor="black")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Count")
        ax.set_title("Missing Values by Column")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        self._save(fig, save_path)
        return fig

    def outlier_boxplot(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot boxplots to visualize outliers in numeric columns.

        Args:
            df: DataFrame containing numeric columns.
            columns: Optional subset of columns to plot; defaults to all numeric.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib figure.
        """
        numeric = df.select_dtypes(include=[np.number])
        selected = columns if columns is not None else list(numeric.columns)
        selected = [col for col in selected if col in numeric.columns]
        fig, ax = plt.subplots(figsize=(10, max(4, len(selected) * 1.2)))
        if not selected:
            ax.text(0.5, 0.5, "No numeric columns to plot", ha="center")
            ax.axis("off")
            self._save(fig, save_path)
            return fig
        data = [numeric[col].dropna().astype(float).values for col in selected]
        ax.boxplot(data, tick_labels=selected, showfliers=True)
        ax.set_title("Outlier Boxplots")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        self._save(fig, save_path)
        return fig

    def _save(self, fig: plt.Figure, save_path: str | Path | None) -> None:
        """Save a figure to disk if a path is provided.

        Args:
            fig: The figure to save.
            save_path: Destination path, or None to skip saving.
        """
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
