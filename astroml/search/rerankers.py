from typing import List, Dict, Any

class Reranker:
    def __init__(self):
        pass

    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # A simple precision reranker: penalizes long documents slightly and boosts exact title token matches
        query_words = set(query.lower().split())
        reranked = []
        for r in results:
            doc = r["document"]
            score = r["score"]
            
            # Title bonus
            title_words = set(doc["title"].lower().split())
            overlap = len(query_words.intersection(title_words))
            bonus = 0.1 * overlap
            
            new_score = score + bonus
            reranked.append({
                "document": doc,
                "score": new_score,
                "method": r["method"]
            })
            
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked
