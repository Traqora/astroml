from typing import List, Dict, Any, Optional
from .retrievers import Retriever
from .rerankers import Reranker
from .indexer import get_indexer

class SearchEngine:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()

    def search(
        self,
        query: str,
        mode: str = "hybrid", # "semantic", "keyword", "hybrid"
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # 1. Retrieve
        if mode == "semantic":
            results = self.retriever.retrieve_semantic(query, top_k=top_k * 2)
        elif mode == "keyword":
            results = self.retriever.retrieve_keyword(query, top_k=top_k * 2)
        else:
            results = self.retriever.retrieve_hybrid(query, top_k=top_k * 2)
            
        # 2. Filter metadata
        if filters:
            filtered_results = []
            for r in results:
                doc = r["document"]
                match = True
                for k, v in filters.items():
                    # Check in document root or metadata
                    doc_val = doc.get(k) or doc.get("metadata", {}).get(k)
                    if doc_val != v:
                        match = False
                        break
                if match:
                    filtered_results.append(r)
            results = filtered_results
            
        # 3. Rerank
        reranked_results = self.reranker.rerank(query, results)
        return reranked_results[:top_k]

_engine = SearchEngine()

def get_search_engine() -> SearchEngine:
    return _engine
