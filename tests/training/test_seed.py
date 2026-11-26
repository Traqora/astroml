"""Tests for the shared deterministic seed setup (issue #1206)."""

from __future__ import annotations

import os
import random
import sys

import pytest

from astroml.training.seed import MAX_SEED, set_seed, validate_seed


def test_validate_seed_accepts_supported_bounds() -> None:
    """The inclusive 32-bit bounds are valid."""
    assert validate_seed(0) == 0
    assert validate_seed(MAX_SEED) == MAX_SEED


@pytest.mark.parametrize("bad_seed", [-1, MAX_SEED + 1, True, 1.5, "42", None])
def test_invalid_seed_is_rejected(bad_seed: object) -> None:
    """Bad values fail before any generator is changed."""
    with pytest.raises(ValueError, match="Seed must"):
        set_seed(bad_seed)  # type: ignore[arg-type]


def test_set_seed_reproduces_all_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-seeding reproduces Python, NumPy and Torch draws and sets cuDNN flags."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_seed(17)
    first = (random.random(), float(np.random.random()), torch.rand(1).item())
    assert os.environ["PYTHONHASHSEED"] == "17"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_seed(17)
    second = (random.random(), float(np.random.random()), torch.rand(1).item())
    assert second == first


def test_set_seed_without_optional_packages_still_seeds_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy and Torch are optional for the utility itself."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    set_seed(23)

    assert random.random() == random.Random(23).random()


def test_seed_range_is_not_silent_about_oversized_values() -> None:
    """The upper bound is stated explicitly in validation errors."""
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        validate_seed(MAX_SEED + 1)
