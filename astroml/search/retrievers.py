import math
from typing import List, Dict, Any, Optional
from .indexer import get_indexer
from .embedders import get_embedder

class Retriever:
    def __init__(self):
        pass

    def retrieve_keyword(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        indexer = get_indexer()
        words = indexer._tokenize(query)
        if not words:
            return []
            
        scores = []
        N = len(indexer.documents)
        
        for idx, doc in enumerate(indexer.documents):
            score = 0.0
            tf_dict = indexer.term_freqs[idx]
            for w in words:
                if w in tf_dict:
                    df = indexer.doc_freqs.get(w, 0)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    tf = tf_dict[w]
                    # Simple BM25-like scoring
                    score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (sum(tf_dict.values()) / 50.0)))
            if score > 0:
                scores.append((score, doc))
                
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"document": d, "score": s, "method": "keyword"} for s, d in scores[:top_k]]

    def retrieve_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        indexer = get_indexer()
        embedder = get_embedder()
        query_vec = embedder.generate_embedding(query)
        
        scores = []
        for idx, doc in enumerate(indexer.documents):
            doc_vec = indexer.embeddings[idx]
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, doc_vec))
            scores.append((dot, doc))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"document": d, "score": s, "method": "semantic"} for s, d in scores[:top_k]]

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # Normalize and combine scores from keyword and semantic retrievals
        k_results = self.retrieve_keyword(query, top_k=top_k * 2)
        s_results = self.retrieve_semantic(query, top_k=top_k * 2)
        
        combined: Dict[str, Dict[str, Any]] = {}
        
        max_k_score = max([r["score"] for r in k_results]) if k_results else 1.0
        max_s_score = max([r["score"] for r in s_results]) if s_results else 1.0
        
        for r in k_results:
            doc_id = r["document"]["id"]
            norm_score = r["score"] / max_k_score
            combined[doc_id] = {
                "document": r["document"],
                "score": norm_score * 0.4, # weight BM25
                "method": "hybrid"
            }
            
        for r in s_results:
            doc_id = r["document"]["id"]
            norm_score = r["score"] / max_s_score
            if doc_id in combined:
                combined[doc_id]["score"] += norm_score * 0.6 # weight semantic
            else:
                combined[doc_id] = {
                    "document": r["document"],
                    "score": norm_score * 0.6,
                    "method": "hybrid"
                }
                
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]
