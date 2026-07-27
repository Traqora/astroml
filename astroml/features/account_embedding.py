import time
from typing import List, Dict, Any
from astroml.storage.vector_store import VectorStore

class AccountEmbeddingGenerator:
    def __init__(self, vector_store: VectorStore, embedding_model=None):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
    def extract_pipeline(self, account_id: str, transactions: List[Dict[str, Any]]) -> str:
        # Simplistic extraction of account features from transaction history
        total_tx = len(transactions)
        total_volume = sum(tx.get('amount', 0) for tx in transactions)
        unique_counterparties = len(set(tx.get('to_address') for tx in transactions if tx.get('to_address')))
        
        text_representation = (
            f"Account {account_id} has {total_tx} transactions with a total volume of {total_volume}. "
            f"Interacted with {unique_counterparties} unique counterparties."
        )
        return text_representation

    def generate_embedding(self, text: str) -> List[float]:
        # Mocking embedding generation
        # In reality, you'd use self.embedding_model.encode(text)
        return [0.1] * 128  # Example dimension

    def process_account(self, account_id: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_repr = self.extract_pipeline(account_id, transactions)
        embedding = self.generate_embedding(text_repr)
        
        return {
            "id": account_id,
            "text": text_repr,
            "vector": embedding,
            "metadata": {
                "tx_count": len(transactions)
            }
        }

    def batch_generate(self, accounts_data: Dict[str, List[Dict[str, Any]]], collection_name: str = "account_embeddings"):
        """
        Process a batch of accounts and store their embeddings in the vector store.
        accounts_data: mapping of account_id to list of transactions
        """
        documents = []
        start_time = time.time()
        
        for account_id, transactions in accounts_data.items():
            doc = self.process_account(account_id, transactions)
            documents.append(doc)
            
        # Store in vector database
        self.vector_store.add_documents(collection_name, documents)
        
        end_time = time.time()
        latency = end_time - start_time
        
        return {
            "processed_count": len(documents),
            "latency_seconds": latency,
            "status": "success" if latency < len(accounts_data) else "warning_slow" # Targeting <1s per account
        }
