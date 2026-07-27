"""Model scorer loading with registry integration (issues #237, #254)."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_scorer_from_path(checkpoint: str):
    """Load InductiveAnomalyScorer from a checkpoint path."""
    try:
        from astroml.pipeline.scoring import InductiveAnomalyScorer  # noqa: PLC0415
        from astroml.pipeline.inductive import InductiveGraphSAGE  # noqa: PLC0415
        from astroml.models.deep_svdd import DeepSVDD  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if not os.path.exists(checkpoint):
            logger.warning("Model checkpoint not found at %s", checkpoint)
            return None

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        input_dim = state.get("input_dim", 8)
        svdd = DeepSVDD(input_dim=input_dim)
        if "svdd_state" in state:
            svdd.load_state_dict(state["svdd_state"])

        from astroml.models.sage_encoder import InductiveSAGEEncoder  # noqa: PLC0415

        encoder = InductiveSAGEEncoder(
            in_channels=input_dim, hidden_channels=64, out_channels=32, num_layers=2
        )
        if "encoder_state" in state:
            encoder.load_state_dict(state["encoder_state"])

        pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[10, 5])
        return InductiveAnomalyScorer(pipeline=pipeline, svdd=svdd)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load scorer from %s: %s", checkpoint, exc)
        return None


async def resolve_active_checkpoint(db: Optional[AsyncSession] = None) -> str:
    """Return the checkpoint path for the active model, or env fallback."""
    default = os.environ.get("MODEL_CHECKPOINT_PATH", "benchmark_results/gcn_model.pt")
    if db is None:
        return default

    try:
        from api.models.orm import ModelRegistry  # noqa: PLC0415

        result = await db.execute(
            select(ModelRegistry)
            .where(ModelRegistry.status == "active")
            .order_by(ModelRegistry.created_at.desc())
            .limit(1)
        )
        active = result.scalar_one_or_none()
        if active and active.path:
            return active.path
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve active model from registry: %s", exc)

    return default


def load_scorer(checkpoint: Optional[str] = None):
    """Load scorer from explicit path or environment default."""
    path = checkpoint or os.environ.get("MODEL_CHECKPOINT_PATH", "benchmark_results/gcn_model.pt")
    return _load_scorer_from_path(path)


def invalidate_scorer_cache() -> None:
    """Clear cached scorer so activation picks up the new checkpoint."""
    _load_scorer_from_path.cache_clear()
