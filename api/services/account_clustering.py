import hdbscan
import numpy as np
from typing import List, Dict, Any

class AccountClustering:
    def __init__(self, embedding_model=None, llm_client=None):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
        self.history = {}

    def cluster_accounts(self, account_embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster accounts using HDBSCAN
        """
        labels = self.clusterer.fit_predict(account_embeddings)
        return labels

    def characterize_cluster(self, cluster_data: List[Dict[str, Any]]) -> str:
        """
        Create cluster characterization with LLM
        """
        if not self.llm_client:
            return "LLM client not configured for characterization"
        
        prompt = f"Analyze these account behaviors and provide a concise 1-sentence label/characterization for this cluster:\n{cluster_data}"
        response = self.llm_client.generate(prompt)
        return response.get("text", "Unknown Cluster Type")

    def track_evolution(self, cluster_id: int, new_data: Dict[str, Any]):
        """
        Track cluster evolution over time
        """
        if cluster_id not in self.history:
            self.history[cluster_id] = []
        self.history[cluster_id].append(new_data)
        
    def get_cluster_summary(self, cluster_id: int) -> Dict[str, Any]:
        return {
            "cluster_id": cluster_id,
            "evolution_history": self.history.get(cluster_id, []),
            "status": "active"
        }
