"""Per-epoch training report, persisted to the model registry (issue #738).

Graph trainers in this package print their progress and then throw it away:
``train_gcn`` prints a validation accuracy every twenty epochs,
``train_link_prediction`` prints an average loss per epoch, and once the
process exits the only surviving record is whatever was in the terminal. The
registry stores a single ``metrics`` blob per model version, so a run's shape —
where it converged, whether validation turned around, which epoch was actually
best — is not recoverable.

:class:`TrainingReport` accumulates per-epoch train/validation metrics, picks
the best epoch under the objective's monitored metric, and writes both the
summary and the history to a registry version.

Where it is written
-------------------
Everything goes into ``ModelVersion.metrics``, the JSON column, under two
keys: flat summary scalars for querying, and ``training_report`` for the
history.

Deliberately *not* ``ModelRegistry.update_model_version_metadata``. There is no
``metadata`` column on ``ModelVersion`` — ``metadata`` is SQLAlchemy's reserved
``MetaData`` attribute on the declarative base, which is why the serving
lineage moved to its own ``lineage`` column in #718. That method still targets
the reserved attribute and cannot persist anything; see the note in
:meth:`TrainingReport.persist`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = ["EpochRecord", "TrainingReport"]


def _clean(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    """Coerce a metric mapping to plain floats.

    Torch scalars and NumPy floats serialise to JSON as opaque objects or not
    at all, and a report whose values cannot be written is worse than no
    report — the failure surfaces at commit time, long after the run.
    """
    if not metrics:
        return {}

    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            cleaned[str(key)] = float(value)
        except (TypeError, ValueError):
            logger.warning("Dropping non-numeric metric %r=%r from training report", key, value)
    return cleaned


@dataclass(frozen=True)
class EpochRecord:
    """One epoch's measurements.

    Attributes:
        epoch: 1-based epoch number.
        train: Metrics measured on the training split.
        val: Metrics measured on the validation split; empty when not evaluated.
        duration_seconds: Wall-clock time for the epoch.
    """

    epoch: int
    train: dict[str, float] = field(default_factory=dict)
    val: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"epoch": self.epoch, "train": dict(self.train)}
        if self.val:
            record["val"] = dict(self.val)
        if self.duration_seconds:
            record["duration_seconds"] = round(self.duration_seconds, 6)
        return record


class TrainingReport:
    """Accumulates per-epoch metrics and persists them to a registry version.

    Args:
        objective: The :class:`~astroml.training.graph_objectives.GraphObjective`
            being trained. Supplies the monitored metric and its direction, so
            the report does not have to be told separately whether a bigger
            number is better — a mismatch there silently records the worst
            epoch as the best.
        monitor: Overrides the objective's monitored metric.
        higher_is_better: Overrides the objective's direction.
        max_epochs_retained: Cap on retained epoch records. A thousand-epoch
            run on a large graph otherwise writes a JSON blob into every
            registry row; beyond the cap the report keeps the first and the
            most recent epochs and records how many were dropped, which is
            what a convergence curve actually needs.
    """

    def __init__(
        self,
        objective: Any | None = None,
        *,
        monitor: str | None = None,
        higher_is_better: bool | None = None,
        max_epochs_retained: int = 500,
    ) -> None:
        if max_epochs_retained < 2:
            raise ValueError("max_epochs_retained must be at least 2")

        resolved_monitor = monitor or getattr(objective, "monitor", None) or "loss"
        if higher_is_better is None:
            higher_is_better = bool(getattr(objective, "higher_is_better", False))

        self.objective_name: str = str(getattr(objective, "name", "unknown"))
        self.monitor: str = resolved_monitor
        self.higher_is_better: bool = higher_is_better
        self.max_epochs_retained = max_epochs_retained

        self._records: list[EpochRecord] = []
        self._dropped = 0
        self._started_at = datetime.now(timezone.utc)

    # -- recording ----------------------------------------------------------

    def record(
        self,
        epoch: int,
        train: Mapping[str, Any] | None = None,
        val: Mapping[str, Any] | None = None,
        duration_seconds: float = 0.0,
    ) -> EpochRecord:
        """Record one epoch and return the stored record."""
        if epoch < 1:
            raise ValueError(f"epoch must be 1-based, got {epoch}")

        record = EpochRecord(
            epoch=epoch,
            train=_clean(train),
            val=_clean(val),
            duration_seconds=float(duration_seconds),
        )
        self._records.append(record)
        self._prune()
        return record

    def _prune(self) -> None:
        """Keep the first epoch and the most recent ones.

        The first epoch is kept deliberately: it is the baseline every later
        number is read against, and dropping it turns a convergence curve into
        an unanchored tail.
        """
        if len(self._records) <= self.max_epochs_retained:
            return

        keep_recent = self.max_epochs_retained - 1
        self._records = [self._records[0]] + self._records[-keep_recent:]
        self._dropped += 1

    # -- reading ------------------------------------------------------------

    @property
    def records(self) -> Sequence[EpochRecord]:
        """Retained epoch records, oldest first."""
        return tuple(self._records)

    @property
    def dropped_epochs(self) -> int:
        """Records discarded to stay under ``max_epochs_retained``."""
        return self._dropped

    def _monitored_value(self, record: EpochRecord) -> float | None:
        """Read the monitored metric, preferring validation over training.

        Validation is what "best" should mean; training is the fallback for a
        loop that never evaluates, where the alternative is no best epoch at
        all.
        """
        if self.monitor in record.val:
            return record.val[self.monitor]
        if self.monitor in record.train:
            return record.train[self.monitor]
        return None

    @property
    def best(self) -> EpochRecord | None:
        """The best retained epoch under :attr:`monitor`, or ``None``.

        Ties resolve to the earliest epoch: reaching a score sooner is the
        better run, and it is the checkpoint an early-stopping loop would have
        kept.
        """
        best_record: EpochRecord | None = None
        best_value: float | None = None

        for record in self._records:
            value = self._monitored_value(record)
            if value is None:
                continue
            if best_value is None or (
                value > best_value if self.higher_is_better else value < best_value
            ):
                best_record, best_value = record, value

        return best_record

    @property
    def latest(self) -> EpochRecord | None:
        """The most recent retained epoch."""
        return self._records[-1] if self._records else None

    def to_dict(self) -> dict[str, Any]:
        """The full report, JSON-serialisable."""
        best = self.best
        return {
            "objective": self.objective_name,
            "monitor": self.monitor,
            "higher_is_better": self.higher_is_better,
            "started_at": self._started_at.isoformat(),
            "epochs_recorded": len(self._records) + self._dropped,
            "epochs_retained": len(self._records),
            "epochs_dropped": self._dropped,
            "best_epoch": best.epoch if best else None,
            "total_duration_seconds": round(
                sum(record.duration_seconds for record in self._records), 6
            ),
            "history": [record.to_dict() for record in self._records],
        }

    def summary_metrics(self) -> dict[str, float]:
        """Flat scalars for the registry's queryable ``metrics`` column.

        Prefixed rather than nested so a caller can compare versions without
        parsing the history: ``best_val_auc``, ``final_train_loss``, and so on.
        """
        summary: dict[str, float] = {}

        best = self.best
        if best is not None:
            summary["best_epoch"] = float(best.epoch)
            for key, value in best.train.items():
                summary[f"best_train_{key}"] = value
            for key, value in best.val.items():
                summary[f"best_val_{key}"] = value

        latest = self.latest
        if latest is not None:
            for key, value in latest.train.items():
                summary[f"final_train_{key}"] = value
            for key, value in latest.val.items():
                summary[f"final_val_{key}"] = value

        return summary

    # -- persistence --------------------------------------------------------

    def persist(self, registry: Any, version_id: int) -> Any:
        """Write the report to a registry model version.

        Both the flat summary and the full history land in
        ``ModelVersion.metrics``. The history is nested under a
        ``training_report`` key so it cannot collide with a caller's own
        metric names.

        This does not call ``update_model_version_metadata``: ``ModelVersion``
        has no ``metadata`` column — the name is taken by SQLAlchemy's
        ``MetaData`` on the declarative base — so that method writes nowhere.
        The same defect was fixed for serving lineage in #718 by giving it a
        real ``lineage`` column; giving the report one would need its own
        migration, and the existing ``metrics`` column already holds JSON.

        Args:
            registry: A :class:`~astroml.tracking.model_registry.ModelRegistry`.
            version_id: Primary key of the model version to update.

        Returns:
            The updated model version, or ``None`` when no such version exists.
        """
        payload: dict[str, Any] = dict(self.summary_metrics())
        payload["training_report"] = self.to_dict()

        updated = registry.update_model_version_metrics(version_id, payload)
        if updated is None:
            logger.warning(
                "Could not persist training report: model version id=%s not found", version_id
            )
        else:
            logger.info(
                "Persisted training report for version id=%s (%d epochs, best epoch %s)",
                version_id,
                len(self._records),
                payload.get("best_epoch"),
            )
        return updated

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        best = self.best
        return (
            f"TrainingReport(objective={self.objective_name!r}, monitor={self.monitor!r}, "
            f"epochs={len(self._records)}, best_epoch={best.epoch if best else None})"
        )
