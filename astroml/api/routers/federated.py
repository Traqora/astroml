from astroml.utils.exceptions import AstroMLError
"""FastAPI router for federated learning orchestration and client coordination."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.training.federated.aggregator import (
    AggregationAlgorithm,
    AggregatorFactory,
    ClientUpdate,
)
from astroml.training.federated.client import DPConfig, FederatedClient
from astroml.training.federated.secure_aggregation import MaskedUpdate
from astroml.training.federated.server import (
    DataVolumeWeightedSelector,
    FederatedServer,
    RandomClientSelector,
    RoundRobinSelector,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/federated", tags=["federated-learning"])

_active_sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    input_dim: int = 10
    algorithm: str = Field(default="fedavg", pattern=r"^(fedavg|fedprox|trimmed_mean|median|krum)$")
    use_secure_aggregation: bool = False
    selection_strategy: str = Field(default="random", pattern=r"^(random|weighted|round_robin)$")
    mu_prox: float = 0.01
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_scale: float = 0.01


class RegisterClientRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    client_id: str
    sample_count: int = 100
    metadata: dict[str, Any] | None = None


class SubmitUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    round_id: int
    weights: dict[str, Any]
    sample_count: int = 100
    loss: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    is_masked: bool = False


class StartRoundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clients_per_round: int | None = None
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 32


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    X: list[list[float]]
    y: list[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/sessions", summary="Create a new federated learning session")
async def create_session(payload: CreateSessionRequest) -> dict[str, Any]:
    """Initialize a federated learning server session."""
    session_id = payload.session_id or f"fl_{uuid.uuid4().hex[:8]}"

    if session_id in _active_sessions:
        raise HTTPException(status_code=400, detail=f"Session '{session_id}' already exists.")

    # Init weights
    weights = {
        "weight": np.zeros((payload.input_dim, 1), dtype=np.float32),
        "bias": np.zeros((1,), dtype=np.float32),
    }

    # Strategy
    strategy = {
        "random": RandomClientSelector(),
        "weighted": DataVolumeWeightedSelector(),
        "round_robin": RoundRobinSelector(),
    }.get(payload.selection_strategy, RandomClientSelector())

    aggregator = AggregatorFactory.create(
        payload.algorithm,
        mu=payload.mu_prox,
    )

    server = FederatedServer(
        initial_weights=weights,
        aggregator=aggregator,
        selection_strategy=strategy,
        use_secure_aggregation=payload.use_secure_aggregation,
    )

    _active_sessions[session_id] = {
        "server": server,
        "client_pool": {},
        "config": payload.model_dump(),
        "created_at": str(np.datetime64("now")),
    }

    return {
        "status": "success",
        "session_id": session_id,
        "algorithm": payload.algorithm,
        "secure_aggregation": payload.use_secure_aggregation,
    }


@router.get("/sessions/{session_id}", summary="Get federated learning session status")
async def get_session(session_id: str) -> dict[str, Any]:
    """Retrieve session metadata, registered clients, and current round."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    sess = _active_sessions[session_id]
    server: FederatedServer = sess["server"]

    return {
        "status": "success",
        "session_id": session_id,
        "current_round": server.current_round,
        "registered_clients": server.list_clients(),
        "client_count": len(server.list_clients()),
        "config": sess["config"],
    }


@router.post("/clients/register", summary="Register client node with session")
async def register_client(payload: RegisterClientRequest) -> dict[str, Any]:
    """Register a decentralized client node with a federated training session."""
    if payload.session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{payload.session_id}' not found.")

    sess = _active_sessions[payload.session_id]
    server: FederatedServer = sess["server"]
    server.register_client(
        client_id=payload.client_id,
        sample_count=payload.sample_count,
        metadata=payload.metadata,
    )

    # Optional local client instance
    cfg = sess["config"]
    dp_cfg = DPConfig(
        enabled=cfg["dp_enabled"],
        clip_norm=cfg["dp_clip_norm"],
        noise_scale=cfg["dp_noise_scale"],
    )
    client_obj = FederatedClient(
        client_id=payload.client_id,
        initial_weights=server.distribute_global_weights(),
        dp_config=dp_cfg,
        mu_prox=cfg["mu_prox"],
    )
    sess["client_pool"][payload.client_id] = client_obj

    return {
        "status": "success",
        "message": f"Client '{payload.client_id}' registered successfully.",
        "session_id": payload.session_id,
    }


@router.get("/sessions/{session_id}/global-model", summary="Get global model weights")
async def get_global_model(session_id: str) -> dict[str, Any]:
    """Download the current global model parameter tensors."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    server: FederatedServer = _active_sessions[session_id]["server"]
    weights = server.distribute_global_weights()

    return {
        "status": "success",
        "session_id": session_id,
        "round_id": server.current_round,
        "weights": {k: v.tolist() for k, v in weights.items()},
    }


@router.post("/sessions/{session_id}/updates", summary="Submit client local update")
async def submit_update(session_id: str, payload: SubmitUpdateRequest) -> dict[str, Any]:
    """Submit local model update (raw or masked) from a client node."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    server: FederatedServer = _active_sessions[session_id]["server"]
    weights_np = {k: np.array(v, dtype=np.float32) for k, v in payload.weights.items()}

    if payload.is_masked:
        server.submit_masked_update(
            MaskedUpdate(
                client_id=payload.client_id,
                masked_weights={k: v.tolist() for k, v in weights_np.items()},
                sample_count=payload.sample_count,
                round_id=payload.round_id,
                metadata=payload.metrics,
            )
        )
    else:
        server.submit_client_update(
            ClientUpdate(
                client_id=payload.client_id,
                weights=weights_np,
                sample_count=payload.sample_count,
                loss=payload.loss,
                metrics=payload.metrics,
                round_id=payload.round_id,
            )
        )

    return {
        "status": "success",
        "message": f"Update received from client '{payload.client_id}'.",
    }


@router.post("/sessions/{session_id}/rounds/start", summary="Start a federated round")
async def start_round(session_id: str, payload: StartRoundRequest) -> dict[str, Any]:
    """Trigger execution of a federated learning training round."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    sess = _active_sessions[session_id]
    server: FederatedServer = sess["server"]
    client_pool = sess["client_pool"]

    if not client_pool:
        # If no internal client pool, select registered client IDs to expect updates
        selected = server.select_clients(count=payload.clients_per_round)
        return {
            "status": "waiting_for_updates",
            "round_id": server.current_round + 1,
            "selected_clients": selected,
        }

    try:
        round_res = server.run_round(
            client_pool=client_pool,
            clients_per_round=payload.clients_per_round,
            learning_rate=payload.learning_rate,
            local_epochs=payload.local_epochs,
            batch_size=payload.batch_size,
        )
        return {
            "status": "success",
            "round_id": round_res.round_id,
            "participating_clients": round_res.participating_clients,
            "global_loss": round_res.global_loss,
            "global_metrics": round_res.global_metrics,
            "duration_seconds": round_res.duration_seconds,
        }
    except AstroMLError as e:
        logger.error("Round execution failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/sessions/{session_id}/history", summary="Get training round history")
async def get_history(session_id: str) -> dict[str, Any]:
    """Retrieve historical loss and metric trajectory across rounds."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    server: FederatedServer = _active_sessions[session_id]["server"]
    return {
        "status": "success",
        "session_id": session_id,
        "history": server.get_training_history(),
    }


@router.post("/sessions/{session_id}/evaluate", summary="Evaluate global model")
async def evaluate_model(session_id: str, payload: EvaluateRequest) -> dict[str, Any]:
    """Evaluate current global model on provided validation features and targets."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    server: FederatedServer = _active_sessions[session_id]["server"]
    X = np.array(payload.X, dtype=np.float32)
    y = np.array(payload.y, dtype=np.float32)

    results = server.evaluate_global_model((X, y))
    return {
        "status": "success",
        "session_id": session_id,
        "round_id": server.current_round,
        "metrics": results,
    }
