"""Ingestion modules for AstroML.

Provides streaming and batch data ingestion from the Stellar Horizon API
into the local PostgreSQL database.
"""
from .service import IngestionService, IngestionResult
from .state import IngestionState, StateStore

__all__ = [
    "config",
    "parsers",
    "stream",
    "IngestionService",
    "IngestionResult",
    "IngestionState",
    "StateStore",
]
