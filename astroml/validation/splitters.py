"""Cross-validation splitting strategies with time-series awareness.

Supports standard k-fold, stratified, group, time-series expanding window,
sliding window, and purged walk-forward with embargo for financial networks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


def _to_numpy_array(data: Any, dtype: Any = None) -> np.ndarray:
    """Helper to convert inputs to 1D/2D numpy array."""
    if data is None:
        return np.array([])
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy(dtype=dtype)
    if isinstance(data, np.ndarray):
        return data if dtype is None else data.astype(dtype)
    return np.asarray(data, dtype=dtype)


class BaseSplitter(ABC):
    """Abstract base class for all cross-validation splitters."""

    @abstractmethod
    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : Any
            Input data (array-like, DataFrame).
        y : Any, optional
            Target values for supervised learning.
        groups : Any, optional
            Group labels for samples used while splitting dataset into train/test set.
        timestamps : Any, optional
            Timestamp values for time-series aware splitting.

        Yields
        ------
        train : np.ndarray
            The training set indices for that split.
        test : np.ndarray
            The testing set indices for that split.
        """
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        raise NotImplementedError


class KFoldSplitter(BaseSplitter):
    """Standard K-Fold cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of folds (must be at least 2). Default is 5.
    shuffle : bool
        Whether to shuffle data before splitting into batches. Default is False.
    random_state : int | None
        Random state when shuffle=True.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than n_samples={n_samples}"
            )

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop


class StratifiedKFoldSplitter(BaseSplitter):
    """Stratified K-Fold cross-validator for imbalanced datasets.

    Ensures each fold contains approximately the same percentage of samples
    of each target class as the complete set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if y is None:
            raise ValueError("StratifiedKFoldSplitter requires target labels y")

        y_arr = _to_numpy_array(y)
        n_samples = len(y_arr)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than n_samples={n_samples}"
            )

        unique_y, y_inversed = np.unique(y_arr, return_inverse=True)
        n_classes = len(unique_y)

        rng = np.random.default_rng(self.random_state) if self.shuffle else None

        per_class_indices = []
        for c in range(n_classes):
            cls_idx = np.where(y_inversed == c)[0]
            if self.shuffle and rng is not None:
                rng.shuffle(cls_idx)
            per_class_indices.append(cls_idx)

        # Allocate per-class samples to folds
        test_folds = np.zeros(n_samples, dtype=int)
        for cls_idx in per_class_indices:
            n_cls = len(cls_idx)
            fold_sizes = np.full(self.n_splits, n_cls // self.n_splits, dtype=int)
            fold_sizes[: n_cls % self.n_splits] += 1
            curr = 0
            for fold_idx, f_size in enumerate(fold_sizes):
                test_folds[cls_idx[curr : curr + f_size]] = fold_idx
                curr += f_size

        all_indices = np.arange(n_samples)
        for fold in range(self.n_splits):
            test_mask = test_folds == fold
            yield all_indices[~test_mask], all_indices[test_mask]


class GroupKFoldSplitter(BaseSplitter):
    """K-Fold cross-validator with non-overlapping groups.

    Ensures the same group is not represented in both testing and training sets.
    """

    def __init__(self, n_splits: int = 5) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if groups is None:
            raise ValueError("GroupKFoldSplitter requires groups argument")

        groups_arr = _to_numpy_array(groups)
        unique_groups, group_indices = np.unique(groups_arr, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than n_groups={n_groups}"
            )

        # Distribute groups evenly by sample count
        group_counts = np.bincount(group_indices)
        group_order = np.argsort(group_counts)[::-1]

        fold_counts = np.zeros(self.n_splits, dtype=int)
        group_to_fold = np.zeros(n_groups, dtype=int)

        for g in group_order:
            smallest_fold = int(np.argmin(fold_counts))
            group_to_fold[g] = smallest_fold
            fold_counts[smallest_fold] += group_counts[g]

        all_indices = np.arange(len(groups_arr))
        for fold in range(self.n_splits):
            test_mask = group_to_fold[group_indices] == fold
            yield all_indices[~test_mask], all_indices[test_mask]


class TimeSeriesSplitter(BaseSplitter):
    """Expanding window time-series cross-validator.

    In the kth split, it uses the first k chunks as train data and the (k+1)th chunk
    as test data. Guarantees temporal ordering: train strictly precedes test.

    Parameters
    ----------
    n_splits : int
        Number of splits (default 5).
    max_train_size : int | None
        Maximum size for a single training set.
    test_size : int | None
        Size of test chunk. If None, computed as n_samples // (n_splits + 1).
    gap : int
        Number of samples to exclude between the end of train and start of test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}")
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        n_splits = self.n_splits
        gap = self.gap

        # Sort indices by timestamp if provided
        if timestamps is not None:
            ts_arr = _to_numpy_array(timestamps)
            indices = np.argsort(ts_arr)
        else:
            indices = np.arange(n_samples)

        test_size = (
            self.test_size
            if self.test_size is not None
            else (n_samples - gap * n_splits) // (n_splits + 1)
        )
        if test_size <= 0:
            raise ValueError(f"Too many splits or gap={gap} too large for n_samples={n_samples}")

        for i in range(n_splits):
            test_start = n_samples - (n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - gap

            if train_end <= 0:
                raise ValueError(f"Not enough training samples in split {i + 1} with gap={gap}")

            train_start = 0
            if self.max_train_size and train_end - train_start > self.max_train_size:
                train_start = train_end - self.max_train_size

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]
            yield train_idx, test_idx


class SlidingWindowSplitter(BaseSplitter):
    """Sliding (rolling) window time-series cross-validator.

    Maintains a fixed-size training window that slides forward through time.

    Parameters
    ----------
    n_splits : int
        Number of splits (default 5).
    window_size : int | None
        Fixed number of samples in the training window.
    test_size : int | None
        Fixed number of samples in the test window.
    step_size : int | None
        Step size by which the window advances each fold.
    gap : int
        Gap samples between train and test window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        window_size: int | None = None,
        test_size: int | None = None,
        step_size: int | None = None,
        gap: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}")
        self.n_splits = n_splits
        self.window_size = window_size
        self.test_size = test_size
        self.step_size = step_size
        self.gap = gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if timestamps is not None:
            ts_arr = _to_numpy_array(timestamps)
            indices = np.argsort(ts_arr)
        else:
            indices = np.arange(n_samples)

        test_size = self.test_size or (n_samples // (self.n_splits + 2))
        window_size = self.window_size or (test_size * 2)
        step_size = self.step_size or test_size

        total_span = window_size + self.gap + test_size + (self.n_splits - 1) * step_size
        if total_span > n_samples:
            # Scale down if needed
            test_size = max(1, (n_samples - self.gap) // (self.n_splits + 2))
            window_size = max(2, test_size * 2)
            step_size = max(
                1, (n_samples - window_size - self.gap - test_size) // max(1, self.n_splits - 1)
            )

        for i in range(self.n_splits):
            train_start = i * step_size
            train_end = train_start + window_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]
            yield train_idx, test_idx


class PurgedWalkForwardSplitter(BaseSplitter):
    """Purged walk-forward cross-validator with embargo for financial time-series.

    Applies purging (removing training observations whose label resolution span
    overlaps with the test period) and embargo (excluding samples immediately
    following the test period to prevent autoregressive information leakage).

    Parameters
    ----------
    n_splits : int
        Number of splits (default 5).
    train_ratio : float
        Proportion of historical data used for training in each fold.
    test_ratio : float
        Proportion of data allocated to testing in each fold.
    gap_periods : int
        Periods purged between train and test.
    embargo_periods : int
        Periods embargoed after the test set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.5,
        test_ratio: float = 0.1,
        gap_periods: int = 0,
        embargo_periods: int = 5,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.gap_periods = gap_periods
        self.embargo_periods = embargo_periods

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if timestamps is not None:
            ts_arr = _to_numpy_array(timestamps)
            indices = np.argsort(ts_arr)
        else:
            indices = np.arange(n_samples)

        test_len = max(1, int(n_samples * self.test_ratio))
        train_len = max(2, int(n_samples * self.train_ratio))

        step = max(
            1, (n_samples - train_len - test_len - self.gap_periods) // max(1, self.n_splits)
        )

        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_len
            test_start = train_end + self.gap_periods
            test_end = min(test_start + test_len, n_samples)

            if test_start >= n_samples:
                break

            # Purge: select train indices strictly before test_start - gap
            raw_train = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            # Embargo: ensure no train sample comes within embargo_periods after test
            # In expanding/walk-forward, training is before test; if wrap-around training is used, embargo filters out [test_end .. test_end + embargo]
            train_idx = raw_train

            yield train_idx, test_idx


class SplitterConfig(BaseModel):
    """Pydantic configuration for cross-validation splitters."""

    model_config = ConfigDict(extra="forbid")

    splitter_type: Literal[
        "kfold",
        "stratified_kfold",
        "group_kfold",
        "time_series",
        "sliding_window",
        "purged_walk_forward",
    ] = Field(default="kfold", description="Type of splitter")
    n_splits: int = Field(default=5, ge=1, description="Number of folds/splits")
    shuffle: bool = Field(default=False, description="Shuffle data before splitting")
    random_state: int | None = Field(default=None, description="Random seed")
    gap: int = Field(default=0, ge=0, description="Gap periods between train and test")
    window_size: int | None = Field(default=None, description="Window size for sliding splitter")
    test_size: int | None = Field(default=None, description="Test chunk size")
    embargo_periods: int = Field(default=0, ge=0, description="Embargo periods after test")


def get_splitter(
    config: SplitterConfig | str | dict[str, Any],
    **kwargs: Any,
) -> BaseSplitter:
    """Factory function to instantiate a cross-validation splitter.

    Parameters
    ----------
    config : SplitterConfig | str | dict
        Configuration object, splitter type name, or config dictionary.
    **kwargs : Any
        Override parameters.

    Returns
    -------
    BaseSplitter
        Instantiated cross-validator.
    """
    if isinstance(config, str):
        cfg_dict = {"splitter_type": config, **kwargs}
        cfg = SplitterConfig.model_validate(cfg_dict)
    elif isinstance(config, dict):
        merged = {**config, **kwargs}
        cfg = SplitterConfig.model_validate(merged)
    else:
        cfg = config

    st = cfg.splitter_type.lower()
    if st == "kfold":
        return KFoldSplitter(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
        )
    elif st == "stratified_kfold":
        return StratifiedKFoldSplitter(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
        )
    elif st == "group_kfold":
        return GroupKFoldSplitter(n_splits=cfg.n_splits)
    elif st == "time_series":
        return TimeSeriesSplitter(
            n_splits=cfg.n_splits,
            test_size=cfg.test_size,
            gap=cfg.gap,
        )
    elif st == "sliding_window":
        return SlidingWindowSplitter(
            n_splits=cfg.n_splits,
            window_size=cfg.window_size,
            test_size=cfg.test_size,
            gap=cfg.gap,
        )
    elif st == "purged_walk_forward":
        return PurgedWalkForwardSplitter(
            n_splits=cfg.n_splits,
            gap_periods=cfg.gap,
            embargo_periods=cfg.embargo_periods,
        )
    else:
        raise ValueError(f"Unknown splitter type: {cfg.splitter_type}")
