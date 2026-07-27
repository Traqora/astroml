"""Natural Language Query Interface Package."""
from __future__ import annotations

from astroml.llm.query.sql_generator import generate_sql
from astroml.llm.query.schema_provider import get_db_schema_context, get_few_shot_examples
from astroml.llm.query.validator import validate_query_safety, QueryValidationError
from astroml.llm.query.executor import execute_safe_query
from astroml.llm.query.pipeline_generator import generate_pipeline_config
from astroml.llm.query.formatter import format_query_results, get_query_suggestions

__all__ = [
    "generate_sql",
    "get_db_schema_context",
    "get_few_shot_examples",
    "validate_query_safety",
    "QueryValidationError",
    "execute_safe_query",
    "generate_pipeline_config",
    "format_query_results",
    "get_query_suggestions",
]
