from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
from astroml.search.engine import get_search_engine
from astroml.search.indexer import get_indexer

router = APIRouter(prefix="/api/v1/search", tags=["search"])

@router.get("")
def search(
    query: str = Query(..., description="Query string"),
    mode: str = Query("hybrid", description="Search mode: semantic, keyword, hybrid"),
    type: Optional[str] = Query(None, description="Filter by data source type"),
    limit: int = Query(10, ge=1, le=100)
):
    try:
        engine = get_search_engine()
        filters = {}
        if type:
            filters["type"] = type
            
        results = engine.search(query=query, mode=mode, top_k=limit, filters=filters if filters else None)
        return {
            "query": query,
            "mode": mode,
            "results": [
                {
                    "id": r["document"]["id"],
                    "title": r["document"]["title"],
                    "content": r["document"]["content"],
                    "type": r["document"]["type"],
                    "score": r["score"],
                    "method": r["method"],
                    "metadata": r["document"].get("metadata", {})
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/autocomplete")
def autocomplete(q: str = Query(..., min_length=1)):
    indexer = get_indexer()
    q_lower = q.lower()
    suggestions = []
    for doc in indexer.documents:
        if q_lower in doc["title"].lower():
            suggestions.append(doc["title"])
    return {"suggestions": list(set(suggestions))[:5]}

@router.post("/reindex")
def reindex():
    try:
        indexer = get_indexer()
        indexer.rebuild_index()
        return {"status": "success", "indexed_count": len(indexer.documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
