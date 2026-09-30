"""Multi-modal preprocessing utilities (issue #631).

Provides data loading, normalization, and batching helpers for
multi-modal datasets (text, image, tabular).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MultiModalSample:
    """A single multi-modal data sample.

    Attributes:
        text: Optional text string.
        image: Optional image array (C, H, W).
        tabular: Optional tabular feature vector.
        label: Optional integer label.
        sample_id: Optional sample identifier.
    """

    text: str | None = None
    image: np.ndarray | None = None
    tabular: np.ndarray | None = None
    label: int | None = None
    sample_id: str | None = None


def collate_batch(
    samples: Sequence[MultiModalSample],
) -> dict[str, Any]:
    """Collate a list of MultiModalSample into a batch dict.

    Args:
        samples: List of samples to collate.

    Returns:
        Dict with keys: text_inputs, image_inputs, tabular_inputs, labels.
    """
    text_inputs: list[str] = []
    image_list: list[np.ndarray] = []
    tabular_list: list[np.ndarray] = []
    labels: list[int] = []

    for s in samples:
        if s.text is not None:
            text_inputs.append(s.text)
        if s.image is not None:
            image_list.append(s.image)
        if s.tabular is not None:
            tabular_list.append(s.tabular)
        if s.label is not None:
            labels.append(s.label)

    return {
        "text_inputs": text_inputs if text_inputs else None,
        "image_inputs": np.stack(image_list) if image_list else None,
        "tabular_inputs": np.stack(tabular_list) if tabular_list else None,
        "labels": np.array(labels, dtype=np.int32) if labels else None,
    }