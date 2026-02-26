"""Validation modules for AstroML.

Expose data integrity and leakage detection utilities here.
"""
# Import validation modules for hash-based deduplication and integrity
from . import dedupe
from . import hashing
from . import integrity
from . import validator

# Try to import leakage (may fail if numpy is not installed)
try:
    from . import leakage
    __all__ = [
        "leakage",
        "dedupe",
        "hashing",
        "validator",
        "integrity",
    ]
except ImportError:
    __all__ = [
        "dedupe",
        "hashing",
        "validator",
        "integrity",
    ]
