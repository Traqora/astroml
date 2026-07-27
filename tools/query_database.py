"""Tool: Execute safe SQL queries."""

from typing import Any
from astroml.llm.tools.definitions import BaseTool


class QueryDatabaseTool(BaseTool):
    name = "query_database"
    description = "Execute safe SQL queries against the AstroML database. Only SELECT queries are allowed."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute (SELECT only)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows to return",
                "default": 100,
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> Any:
        query = params["query"].strip().upper()
        if not query.startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed"}
        limit = params.get("limit", 100)
        return {"rows": [], "row_count": 0, "limit": limit, "note": "Query execution not connected — stub result"}
