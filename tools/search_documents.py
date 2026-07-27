"""Tool: Semantic search in documentation."""

from typing import Any
from astroml.llm.tools.definitions import BaseTool


class SearchDocumentsTool(BaseTool):
    name = "search_documents"
    description = "Perform semantic search across AstroML documentation and knowledge base."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> Any:
        query = params["query"]
        top_k = params.get("top_k", 5)
        return {
            "query": query,
            "results": [
                {"title": "AstroML Overview", "snippet": "AstroML is a fraud detection platform...", "score": 0.95},
                {"title": "Getting Started", "snippet": "Install astroml with pip install astroml...", "score": 0.88},
            ],
            "total_results": 2,
            "top_k": top_k,
            "note": "Search results are sample data — connect to document store for live results",
        }
