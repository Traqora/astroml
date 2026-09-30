"""Pytest configuration for the AstroML test suite.

Ensures the repository root is importable so ``import astroml`` resolves when
pytest is invoked as a console script (``pytest -v``) from the project root,
without requiring ``pip install -e .`` first.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
