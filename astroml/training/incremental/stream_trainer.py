"""Stream data ingestion and online training pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterable, Iterator, Sequence

from .adaptive_model import AdaptiveModel, AdaptiveModelConfig


@dataclass
class StreamTrainerConfig:
    """Configuration for stream trainer."""

    batch_size: int = 32
    max_samples: int | None = None
    log_interval_batches: int = 10
    eval_metric: str = "accuracy"  # "accuracy", "mse", "f1"
    classes: list[int] = field(default_factory=lambda: [0, 1])


class StreamDataIngestor:
    """Data ingestor for streaming continuous updates."""

    @staticmethod
    def batch_stream(
        data_stream: Iterable[tuple[Sequence[float], Any]],
        batch_size: int = 32,
    ) -> Iterator[tuple[list[list[float]], list[Any]]]:
        """Yield batches from an unbounded or bounded data stream."""
        batch_X: list[list[float]] = []
        batch_y: list[Any] = []

        for x_i, y_i in data_stream:
            batch_X.append(list(x_i))
            batch_y.append(y_i)
            if len(batch_X) >= batch_size:
                yield batch_X, batch_y
                batch_X = []
                batch_y = []

        if batch_X:
            yield batch_X, batch_y


class StreamTrainer:
    """Trainer coordinating online streaming ingestion, prequential evaluation, and updates."""

    def __init__(
        self,
        model: AdaptiveModel,
        config: StreamTrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or StreamTrainerConfig()
        self.metrics_history: list[dict[str, Any]] = []
        self.total_samples_trained: int = 0

    def calculate_metrics(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
        """Compute comprehensive evaluation metrics on streaming evaluation chunk."""
        if not y_true or not y_pred:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "error_rate": 0.0}

        n = len(y_true)
        # Check classification or regression
        is_classification = all(isinstance(y, (int, bool)) or (isinstance(y, float) and y in (0.0, 1.0)) for y in y_true)

        if is_classification:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

            accuracy = (tp + tn) / max(1, n)
            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            f1 = 2 * precision * recall / max(1e-12, (precision + recall))
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "error_rate": 1.0 - accuracy,
            }
        else:
            # Regression metrics: MSE, MAE, RMSE
            errors = [abs(float(yt) - float(yp)) for yt, yp in zip(y_true, y_pred)]
            sq_errors = [(float(yt) - float(yp)) ** 2 for yt, yp in zip(y_true, y_pred)]
            mae = sum(errors) / max(1, n)
            mse = sum(sq_errors) / max(1, n)
            return {
                "mae": mae,
                "mse": mse,
                "rmse": mse ** 0.5,
                "error_rate": mae,
            }

    def train_stream(
        self,
        data_stream: Iterable[tuple[Sequence[float], Any]],
        on_batch_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train continuously on a data stream using prequential evaluation (test-then-train)."""
        batches = StreamDataIngestor.batch_stream(data_stream, batch_size=self.config.batch_size)
        batch_idx = 0
        all_true: list[Any] = []
        all_pred: list[Any] = []
        start_time = time.time()

        for batch_X, batch_y in batches:
            if self.config.max_samples and self.total_samples_trained >= self.config.max_samples:
                break

            # 1. Test (prequential evaluation)
            preds = self.model.predict(batch_X)
            all_true.extend(batch_y)
            all_pred.extend(preds)

            batch_metrics = self.calculate_metrics(batch_y, preds)

            # 2. Train (incremental update)
            adapt_info = self.model.adapt_and_update(batch_X, batch_y, classes=self.config.classes)

            self.total_samples_trained += len(batch_X)
            batch_idx += 1

            record = {
                "batch_index": batch_idx,
                "samples_processed": self.total_samples_trained,
                "metrics": batch_metrics,
                "adapt_info": adapt_info,
                "timestamp": time.time(),
            }
            self.metrics_history.append(record)

            if on_batch_complete is not None:
                on_batch_complete(record)

        cumulative_metrics = self.calculate_metrics(all_true, all_pred)
        elapsed = time.time() - start_time

        return {
            "total_batches": batch_idx,
            "total_samples": self.total_samples_trained,
            "cumulative_metrics": cumulative_metrics,
            "elapsed_seconds": elapsed,
            "metrics_history": self.metrics_history,
        }
