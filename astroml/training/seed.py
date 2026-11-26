"""Deterministic random seed setup for the supported v1 training path.

The seed is applied to Python's ``random`` module, NumPy, PyTorch (CPU and all
CUDA devices) and the cuDNN deterministic flags. ``PYTHONHASHSEED`` is exported
for child processes. NumPy and PyTorch are optional: if either is unavailable
the remaining generators are still seeded. Reproducibility is guaranteed only
for the same hardware, library versions and seed; different devices may still
use different kernels.
"""

from __future__ import annotations

import os
import random
from typing import Final

MIN_SEED: Final[int] = 0
MAX_SEED: Final[int] = 2**32 - 1

__all__ = ["MIN_SEED", "MAX_SEED", "validate_seed", "set_seed"]


def validate_seed(seed: int) -> int:
    """Return ``seed`` when it is an integer in the supported 32-bit range."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer in [{MIN_SEED}, {MAX_SEED}], got {seed!r}.")
    if not MIN_SEED <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be in [{MIN_SEED}, {MAX_SEED}], got {seed!r}.")
    return seed


def set_seed(seed: int) -> None:
    """Seed every random generator used by v1 training.

    Args:
        seed: Integer in ``[0, 2**32 - 1]``. NaN, booleans and out-of-range
            values raise :class:`ValueError` before any generator is touched.

    Raises:
        ValueError: ``seed`` is not a supported integer.
    """
    seed = validate_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
