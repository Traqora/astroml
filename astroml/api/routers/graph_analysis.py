"""Graph Analysis API Router for AstroML.

Provides endpoints for graph topology metrics, GNN node classification,
fraud risk scoring, embedding extraction, and subgraph analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from astroml.graph_utils import (
    build_transaction_pyg_graph,
    compute_graph_metrics,
    extract_k_hop_subgraph,
    generate_node_embeddings,
)
from astroml.training.gnn.data_loader import create_node_masks
from astroml.training.gnn.gat import GATNodeClassifier, train_gat_classifier
from astroml.training.gnn.graph_sage import (
    GraphSAGENodeClassifier,
    train_graphsage_classifier,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graph-analysis", tags=["graph-analysis"])


# ---------------------------------------------------------------------------
# Request and Response Models
# ---------------------------------------------------------------------------


class TransactionRecord(BaseModel):
    source_account: str
    destination_account: str
    amount: float = 1.0
    fee: float = 0.0
    asset: str = "XLM"
    edge_type: int = 0
    timestamp: str | None = None


class GraphAnalysisRequest(BaseModel):
    transactions: list[TransactionRecord] = Field(..., description="List of transactions")
    directed: bool = Field(default=True, description="Whether the graph is directed")


class GraphAnalysisResponse(BaseModel):
    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    max_degree: int
    min_degree: int
    avg_in_degree: float
    avg_out_degree: float
    isolated_nodes_count: int


class NodeClassificationRequest(BaseModel):
    transactions: list[TransactionRecord]
    model_type: str = Field(default="graphsage", description="'graphsage' or 'gat'")
    hidden_dim: int = Field(default=32, ge=4, le=512)
    num_classes: int = Field(default=2, ge=2)
    node_features: dict[str, list[float]] | None = None
    labels: dict[str, int] | None = None


class NodePrediction(BaseModel):
    account: str
    predicted_class: int
    fraud_probability: float
    risk_score: float


class NodeClassificationResponse(BaseModel):
    model_type: str
    predictions: list[NodePrediction]
    num_nodes: int
    num_edges: int


class NodeEmbeddingRequest(BaseModel):
    transactions: list[TransactionRecord]
    model_type: str = Field(default="graphsage", description="'graphsage' or 'gat'")
    embedding_dim: int = Field(default=32, ge=4, le=256)
    node_features: dict[str, list[float]] | None = None


class NodeEmbeddingResponse(BaseModel):
    model_type: str
    embedding_dim: int
    embeddings: dict[str, list[float]]


class SubgraphRequest(BaseModel):
    transactions: list[TransactionRecord]
    target_accounts: list[str]
    num_hops: int = Field(default=2, ge=1, le=5)


class SubgraphResponse(BaseModel):
    target_accounts: list[str]
    subgraph_nodes: list[str]
    num_nodes: int
    num_edges: int


class GNNTrainRequest(BaseModel):
    transactions: list[TransactionRecord]
    labels: dict[str, int] = Field(..., description="Ground truth labels for accounts")
    model_type: str = Field(default="graphsage", description="'graphsage' or 'gat'")
    epochs: int = Field(default=50, ge=1, le=500)
    lr: float = Field(default=0.01, gt=0.0)
    hidden_dim: int = Field(default=32, ge=8)


class GNNTrainResponse(BaseModel):
    status: str
    model_type: str
    epochs_trained: int
    final_train_loss: float
    final_train_acc: float
    final_val_acc: float | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Health check for graph analysis service."""
    return {"status": "ok", "service": "graph-analysis"}


@router.post("/analyze", response_model=GraphAnalysisResponse)
async def analyze_graph(request: GraphAnalysisRequest) -> GraphAnalysisResponse:
    """Analyze topology and statistics of a transaction graph."""
    if not request.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction list cannot be empty",
        )

    tx_dicts = [t.model_dump() for t in request.transactions]
    data, _ = build_transaction_pyg_graph(tx_dicts)
    metrics = compute_graph_metrics(data)
    return GraphAnalysisResponse(**metrics)


@router.post("/classify-nodes", response_model=NodeClassificationResponse)
async def classify_nodes(request: NodeClassificationRequest) -> NodeClassificationResponse:
    """Classify nodes in transaction graph for fraud risk scoring."""
    if not request.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction list cannot be empty",
        )

    tx_dicts = [t.model_dump() for t in request.transactions]
    data, node_to_id = build_transaction_pyg_graph(
        transactions=tx_dicts,
        node_feature_dict=request.node_features,
        label_dict=request.labels,
    )

    if data.num_nodes == 0:
        return NodeClassificationResponse(
            model_type=request.model_type,
            predictions=[],
            num_nodes=0,
            num_edges=0,
        )

    in_dim = data.x.size(1)
    model_type = request.model_type.lower()

    if model_type == "gat":
        model = GATNodeClassifier(
            in_channels=in_dim,
            hidden_channels=request.hidden_dim,
            out_channels=request.num_classes,
            heads=2,
            out_heads=1,
            num_layers=2,
        )
    else:
        model = GraphSAGENodeClassifier(
            in_channels=in_dim,
            hidden_channels=request.hidden_dim,
            out_channels=request.num_classes,
            num_layers=2,
        )

    probs = model.predict_proba(data.x, data.edge_index).cpu().numpy()
    preds = model.predict(data.x, data.edge_index).cpu().numpy()

    id_to_node = {idx: acc for acc, idx in node_to_id.items()}
    predictions = []

    for idx in range(data.num_nodes):
        acc = id_to_node.get(idx, f"node_{idx}")
        prob_fraud = float(probs[idx][1]) if request.num_classes >= 2 else float(probs[idx][0])
        predictions.append(
            NodePrediction(
                account=acc,
                predicted_class=int(preds[idx]),
                fraud_probability=prob_fraud,
                risk_score=float(prob_fraud * 100.0),
            )
        )

    return NodeClassificationResponse(
        model_type=request.model_type,
        predictions=predictions,
        num_nodes=data.num_nodes,
        num_edges=data.edge_index.size(1),
    )


@router.post("/embeddings", response_model=NodeEmbeddingResponse)
async def extract_embeddings(request: NodeEmbeddingRequest) -> NodeEmbeddingResponse:
    """Generate node embeddings from transaction network."""
    if not request.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction list cannot be empty",
        )

    tx_dicts = [t.model_dump() for t in request.transactions]
    data, node_to_id = build_transaction_pyg_graph(
        transactions=tx_dicts,
        node_feature_dict=request.node_features,
    )

    if data.num_nodes == 0:
        return NodeEmbeddingResponse(
            model_type=request.model_type,
            embedding_dim=request.embedding_dim,
            embeddings={},
        )

    in_dim = data.x.size(1)
    model_type = request.model_type.lower()

    if model_type == "gat":
        model = GATNodeClassifier(
            in_channels=in_dim,
            hidden_channels=request.embedding_dim,
            out_channels=request.embedding_dim,
            heads=2,
            out_heads=1,
            num_layers=2,
        )
    else:
        model = GraphSAGENodeClassifier(
            in_channels=in_dim,
            hidden_channels=request.embedding_dim,
            out_channels=request.embedding_dim,
            num_layers=2,
        )

    embeddings_np = generate_node_embeddings(model, data)
    id_to_node = {idx: acc for acc, idx in node_to_id.items()}

    res_embeddings = {id_to_node[idx]: embeddings_np[idx].tolist() for idx in range(data.num_nodes)}

    return NodeEmbeddingResponse(
        model_type=request.model_type,
        embedding_dim=embeddings_np.shape[1],
        embeddings=res_embeddings,
    )


@router.post("/subgraph", response_model=SubgraphResponse)
async def extract_subgraph_endpoint(request: SubgraphRequest) -> SubgraphResponse:
    """Extract ego subgraph around requested accounts."""
    if not request.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction list cannot be empty",
        )

    tx_dicts = [t.model_dump() for t in request.transactions]
    data, node_to_id = build_transaction_pyg_graph(tx_dicts)

    target_indices = [node_to_id[acc] for acc in request.target_accounts if acc in node_to_id]

    if not target_indices:
        return SubgraphResponse(
            target_accounts=request.target_accounts,
            subgraph_nodes=[],
            num_nodes=0,
            num_edges=0,
        )

    subset_nodes, sub_edge_index, _, _ = extract_k_hop_subgraph(
        node_idx=target_indices,
        num_hops=request.num_hops,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
    )

    id_to_node = {idx: acc for acc, idx in node_to_id.items()}
    subgraph_node_names = [id_to_node[idx.item()] for idx in subset_nodes]

    return SubgraphResponse(
        target_accounts=request.target_accounts,
        subgraph_nodes=subgraph_node_names,
        num_nodes=len(subgraph_node_names),
        num_edges=sub_edge_index.size(1),
    )


@router.post("/train", response_model=GNNTrainResponse)
async def train_gnn_endpoint(request: GNNTrainRequest) -> GNNTrainResponse:
    """Train a GraphSAGE or GAT model on provided transaction dataset."""
    if not request.transactions or not request.labels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transactions and labels are required",
        )

    tx_dicts = [t.model_dump() for t in request.transactions]
    data, node_to_id = build_transaction_pyg_graph(
        transactions=tx_dicts,
        label_dict=request.labels,
    )

    if data.num_nodes == 0 or data.y is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid nodes with labels found in transaction graph",
        )

    train_mask, val_mask, _ = create_node_masks(
        num_nodes=data.num_nodes,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        labels=data.y,
        stratify=False,
    )

    in_dim = data.x.size(1)
    num_classes = len(torch.unique(data.y))
    model_type = request.model_type.lower()

    if model_type == "gat":
        model = GATNodeClassifier(
            in_channels=in_dim,
            hidden_channels=request.hidden_dim,
            out_channels=max(2, num_classes),
            heads=2,
            out_heads=1,
            num_layers=2,
        )
        res = train_gat_classifier(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            labels=data.y,
            train_mask=train_mask,
            val_mask=val_mask,
            epochs=request.epochs,
            lr=request.lr,
        )
    else:
        model = GraphSAGENodeClassifier(
            in_channels=in_dim,
            hidden_channels=request.hidden_dim,
            out_channels=max(2, num_classes),
            num_layers=2,
        )
        res = train_graphsage_classifier(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            labels=data.y,
            train_mask=train_mask,
            val_mask=val_mask,
            epochs=request.epochs,
            lr=request.lr,
        )

    return GNNTrainResponse(
        status="success",
        model_type=request.model_type,
        epochs_trained=res["epochs_trained"],
        final_train_loss=res["final_train_loss"],
        final_train_acc=res["final_train_acc"],
        final_val_acc=res["final_val_acc"],
    )
