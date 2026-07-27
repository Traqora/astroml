from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from ..services.account_clustering import AccountClustering
from ..services.rag_system import BlockchainRAGSystem

router = APIRouter()

# Dependency injections (mocked for now)
def get_clustering_service():
    return AccountClustering()

def get_rag_system():
    return BlockchainRAGSystem()

class QARequest(BaseModel):
    query: str
    session_id: str = "default"

@router.get("/api/v1/accounts/clusters")
def get_account_clusters(service: AccountClustering = Depends(get_clustering_service)):
    """
    Get cluster characterizations and tracking.
    """
    # This is a mock response to satisfy the endpoint requirement
    return {
        "status": "success",
        "clusters": [
            service.get_cluster_summary(0),
            service.get_cluster_summary(1)
        ]
    }

@router.post("/api/v1/llm/qa")
def ask_blockchain_question(request: QARequest, service: BlockchainRAGSystem = Depends(get_rag_system)):
    """
    RAG system answering blockchain questions with citations.
    """
    try:
        response = service.answer_question(request.query, request.session_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
