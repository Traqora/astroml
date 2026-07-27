"""Safety validation for generated SQL queries."""
from __future__ import annotations

import re
from typing import Tuple, List

# Banned SQL keywords for read-only safety
BANNED_KEYWORDS = [
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "truncate",
    "grant",
    "revoke",
    "replace",
    "create table",
]


class QueryValidationError(Exception):
    """Exception raised when generated query fails safety checks."""
    pass


def validate_query_safety(sql: str) -> Tuple[bool, List[str]]:
    """
    Validate that the SQL query is read-only and free of SQL injection hazards.
    Returns (is_safe, list_of_violations).
    """
    violations = []
    sql_lower = sql.lower()
    
    # 1. Check for banned keywords
    for keyword in BANNED_KEYWORDS:
        # Match word boundaries to prevent false positives (e.g. "updated_at")
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, sql_lower):
            violations.append(f"Unauthorized action: use of keyword '{keyword}' is banned.")
            
    # 2. Check for SQL Injection patterns (like semicolons followed by other commands)
    if ";" in sql[:-1]:
        violations.append("Multiple statements separated by semicolon are forbidden.")
        
    # 3. Check for complexity (restrict JOIN count to maximum 3)
    join_count = len(re.findall(r"\bjoin\b", sql_lower))
    if join_count > 3:
        violations.append(f"Query too complex: maximum of 3 JOINs allowed (got {join_count}).")
        
    return len(violations) == 0, violations
