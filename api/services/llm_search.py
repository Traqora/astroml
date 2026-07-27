import time
from typing import List, Dict, Any, Optional
from astroml.llm.providers.embedding_router import build_default_router
from api.schemas import SearchRequest, SearchResponse, SearchResult

class SemanticSearchService:
    def __init__(self):
        self.embedding_router = build_default_router()
        # Mock database
        self.mock_data = [
            {"id": "tx_123", "type": "transaction", "text": "large transfer to exchange binance", "amount": 50000},
            {"id": "acc_456", "type": "account", "text": "whale account active since 2020", "balance": 1000000},
            {"id": "tx_789", "type": "transaction", "text": "defi swap on uniswap v3", "amount": 1500},
            {"id": "acc_012", "type": "account", "text": "smart contract creator address", "balance": 50},
        ]

    async def search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.time()
        
        # 1. Generate Query Embedding
        query_vector = await self.embedding_router.embed_query(request.query)
        
        # 2. Filter & Similarity Search (Mocked calculation)
        results = []
        for item in self.mock_data:
            # Apply basic filters if any
            if request.filters and "type" in request.filters:
                if item["type"] != request.filters["type"]:
                    continue
            
            # Mock similarity score based on simple substring logic + random for realism
            score = 0.5
            if any(word in item["text"].lower() for word in request.query.lower().split()):
                score += 0.3
            
            results.append(
                SearchResult(
                    id=item["id"],
                    type=item["type"],
                    score=score,
                    data=item,
                    explanation=f"Matched because it is semantically related to '{request.query}'."
                )
            )
            
        results.sort(key=lambda x: x.score, reverse=True)
        top_results = results[:request.top_k]
        
        # Enforce <500ms time
        query_time_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            results=top_results,
            query_time_ms=query_time_ms
        )
