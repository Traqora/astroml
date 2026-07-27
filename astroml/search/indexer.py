import math
import re
from typing import List, Dict, Any
from .embedders import get_embedder
from .sources.connectors import DocsConnector, TransactionsConnector, AlertsConnector, ModelsConnector, CodeConnector

class Indexer:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.vocab: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        
    def rebuild_index(self):
        self.documents = []
        connectors = [
            DocsConnector(),
            TransactionsConnector(),
            AlertsConnector(),
            ModelsConnector(),
            CodeConnector()
        ]
        
        for conn in connectors:
            self.documents.extend(conn.fetch_documents())
            
        # Recompute embeddings
        embedder = get_embedder()
        texts_to_embed = [f"{d['title']} {d['content']}" for d in self.documents]
        self.embeddings = embedder.generate_embeddings(texts_to_embed)
        
        # Build BM25 / TF-IDF structures
        self.vocab = {}
        self.doc_freqs = {}
        self.term_freqs = []
        
        for doc in self.documents:
            words = self._tokenize(f"{doc['title']} {doc['content']}")
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            self.term_freqs.append(tf)
            
            for w in set(words):
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

_indexer = Indexer()
# Populate initially
_indexer.rebuild_index()

def get_indexer() -> Indexer:
    return _indexer
