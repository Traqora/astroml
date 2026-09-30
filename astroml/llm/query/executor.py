from astroml.utils.exceptions import AstroMLError
"""Safe SQL query executor with timeouts and audit logging."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from astroml.llm.query.validator import QueryValidationError, validate_query_safety

logger = logging.getLogger(__name__)


async def execute_safe_query(
    db: AsyncSession,
    sql: str,
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Execute SQL query safely after validation.
    Enforces read-only timeout (max 30s) and logs execution.
    """
    # 1. Validate query safety
    is_safe, violations = validate_query_safety(sql)
    if not is_safe:
        logger.warning(
            "Audit Log: Blocked query validation failure: '%s'. Violations: %s", sql, violations
        )
        raise QueryValidationError(f"Query validation failed: {', '.join(violations)}")

    logger.info("Audit Log: Executing validated query: '%s'", sql)

    # 2. Run with timeout
    try:
        async with asyncio.timeout(timeout_seconds):
            # Execute raw SQL select
            result = await db.execute(text(sql))

            # Form row dictionaries
            rows = []
            if result.returns_rows:
                for row in result.fetchall():
                    rows.append(dict(row._mapping))
            return rows

    except asyncio.TimeoutError:
        logger.error("Audit Log: Query timed out after %s seconds: '%s'", timeout_seconds, sql)
        raise TimeoutError(f"Query execution timed out after {timeout_seconds}s limit.")
    except AstroMLError as e:
        logger.error("Audit Log: Database execution error for query '%s': %s", sql, e)
        raise e
