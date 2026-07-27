"""Result formatter and suggestions generator for natural language queries."""
from __future__ import annotations

import json
from typing import Dict, Any, List, Optional


def format_query_results(
    results: List[Dict[str, Any]],
    max_rows: int = 10,
) -> Dict[str, Any]:
    """Format and summarize query results to fit within LLM limits/readable layouts."""
    total_count = len(results)
    truncated = total_count > max_rows
    display_rows = results[:max_rows]
    
    # 1. Simple markdown table representation
    markdown_table = ""
    if total_count > 0:
        headers = list(results[0].keys())
        markdown_table += "| " + " | ".join(headers) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in display_rows:
            markdown_table += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
            
    return {
        "total_rows": total_count,
        "returned_rows": len(display_rows),
        "truncated": truncated,
        "markdown_table": markdown_table,
        "raw_json": display_rows
    }


def get_query_suggestions(user_context: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate suggestions based on context (e.g. current page or past operations)."""
    return [
        "Show me all accounts with balance > 1000 XLM in the last 7 days",
        "Top 10 accounts by transaction volume this month",
        "Show fraud alerts for accounts with risk score > 0.8",
        "Train a fraud detection model using transactions from last 30 days"
    ]
