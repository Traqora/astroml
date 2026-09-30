"""Model security API: adversarial testing, extraction and poisoning detection.

Resolves part of #645.  Mounted at ``/api/v1/model-security``.

Models are supplied to this router by registering a ``predict_proba`` callable
under a name via :func:`register_model`; endpoints then reference that name.
This keeps arbitrary model loading out of the request path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from astroml.security.adversarial.attacks import AttackConfig, AttackType, generate_attack
from astroml.security.adversarial.defenses import (
    AdversarialDetector,
    FeatureSqueezing,
    GaussianSmoothing,
    RobustnessEvaluator,
)
from astroml.security.model_extraction import ModelExtractionDetector
from astroml.security.poisoning_detection import PoisoningDetector
from astroml.security.scoring import SecurityTestPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/model-security", tags=["model-security"])

PredictProbaFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_MODELS: dict[str, PredictProbaFn] = {}
_extraction_detector = ModelExtractionDetector()


def register_model(name: str, predict_proba: PredictProbaFn) -> None:
    """Make a scoring callable available to this router under ``name``."""
    if not name:
        raise ValueError("model name must not be empty")
    _MODELS[name] = predict_proba


def unregister_model(name: str) -> bool:
    """Remove a registered model; return whether it existed."""
    return _MODELS.pop(name, None) is not None


def registered_models() -> list[str]:
    """Return the names of every registered model."""
    return sorted(_MODELS)


def get_extraction_detector() -> ModelExtractionDetector:
    """Return the extraction detector backing this router."""
    return _extraction_detector


def _resolve_model(name: str) -> PredictProbaFn:
    """Return the registered model, or raise a 404."""
    model = _MODELS.get(name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"model {name!r} is not registered; known models: {registered_models()}",
        )
    return model


# ─── Schemas ─────────────────────────────────────────────────────────────────


class AttackConfigRequest(BaseModel):
    """Attack hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    epsilon: float = Field(default=0.1, gt=0.0)
    step_size: float | None = Field(default=None, gt=0.0)
    max_iterations: int = Field(default=40, ge=1, le=1000)
    targeted: bool = False
    clip_low: float = 0.0
    clip_high: float = 1.0
    random_start: bool = True
    seed: int | None = 0

    def to_config(self) -> AttackConfig:
        """Convert the request into an :class:`AttackConfig`."""
        return AttackConfig(
            epsilon=self.epsilon,
            step_size=self.step_size,
            max_iterations=self.max_iterations,
            targeted=self.targeted,
            clip_range=(self.clip_low, self.clip_high),
            random_start=self.random_start,
            seed=self.seed,
        )


class AttackRequest(BaseModel):
    """Run one attack against a registered model."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    attack: AttackType = AttackType.PGD
    features: list[list[float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)
    config: AttackConfigRequest = Field(default_factory=AttackConfigRequest)
    include_examples: bool = False


class RobustnessRequest(BaseModel):
    """Evaluate a model against several attacks."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    features: list[list[float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)
    attacks: list[AttackType] = Field(default_factory=lambda: [AttackType.FGSM, AttackType.PGD])
    config: AttackConfigRequest = Field(default_factory=AttackConfigRequest)


class DefenseRequest(BaseModel):
    """Screen inputs for adversarial manipulation."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    features: list[list[float]] = Field(min_length=1)
    defense: Literal["feature_squeezing", "gaussian_smoothing"] = "feature_squeezing"
    threshold: float = Field(default=0.3, gt=0.0, le=2.0)


class QueryObservationRequest(BaseModel):
    """Report a prediction request for extraction monitoring."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1)
    features: list[float] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class PoisoningRequest(BaseModel):
    """Screen a training set for poisoning."""

    model_config = ConfigDict(extra="forbid")

    features: list[list[float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)
    n_neighbors: int = Field(default=5, ge=1, le=100)
    outlier_z_threshold: float = Field(default=4.0, gt=0.0)


class SecurityScanRequest(BaseModel):
    """Run the full security suite and score the model."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    eval_features: list[list[float]] | None = None
    eval_labels: list[int] | None = None
    train_features: list[list[float]] | None = None
    train_labels: list[int] | None = None
    config: AttackConfigRequest = Field(default_factory=AttackConfigRequest)
    include_markdown: bool = False


def _as_matrix(rows: list[list[float]]) -> NDArray[np.float64]:
    """Convert a JSON matrix into a validated 2-D float array."""
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        raise HTTPException(status_code=422, detail="features must be a 2-D array")
    return array


def _check_lengths(features: NDArray[np.float64], labels: list[int]) -> NDArray[np.int_]:
    """Validate that labels line up with features and return them as an array."""
    if features.shape[0] != len(labels):
        raise HTTPException(status_code=422, detail="features and labels must have the same length")
    return np.asarray(labels, dtype=int)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Return the models registered for security testing."""
    return {"models": registered_models()}


@router.post("/attack", status_code=200)
async def run_attack(request: AttackRequest) -> dict[str, Any]:
    """Generate adversarial examples for a registered model."""
    model = _resolve_model(request.model_name)
    features = _as_matrix(request.features)
    labels = _check_lengths(features, request.labels)

    result = generate_attack(
        request.attack, model, features, labels, config=request.config.to_config()
    )
    payload = result.to_dict()
    payload["model_name"] = request.model_name
    if request.include_examples:
        payload["adversarial_examples"] = result.adversarial_examples.tolist()
        payload["adversarial_predictions"] = result.adversarial_predictions.tolist()
    return payload


@router.post("/robustness")
async def evaluate_robustness(request: RobustnessRequest) -> dict[str, Any]:
    """Evaluate a model's robustness across several attacks."""
    model = _resolve_model(request.model_name)
    features = _as_matrix(request.features)
    labels = _check_lengths(features, request.labels)

    evaluator = RobustnessEvaluator(
        model, config=request.config.to_config(), attacks=request.attacks
    )
    report = evaluator.evaluate(features, labels)
    return {"model_name": request.model_name, **report.to_dict()}


@router.post("/detect-adversarial")
async def detect_adversarial(request: DefenseRequest) -> dict[str, Any]:
    """Flag inputs that look adversarially perturbed."""
    model = _resolve_model(request.model_name)
    features = _as_matrix(request.features)

    defenses = (
        (FeatureSqueezing(bit_depth=4), FeatureSqueezing(bit_depth=2))
        if request.defense == "feature_squeezing"
        else (GaussianSmoothing(),)
    )
    detector = AdversarialDetector(model, defenses, threshold=request.threshold)
    result = detector.detect(features)
    return {
        "model_name": request.model_name,
        "defense": request.defense,
        **result.to_dict(),
        "flagged_indices": np.flatnonzero(result.is_adversarial).tolist(),
    }


@router.post("/extraction/observe", status_code=202)
async def observe_query(request: QueryObservationRequest) -> dict[str, str]:
    """Record a prediction request for model-extraction monitoring."""
    get_extraction_detector().observe(
        request.client_id, np.asarray(request.features), request.confidence
    )
    return {"status": "accepted"}


@router.get("/extraction/{client_id}")
async def extraction_verdict(client_id: str) -> dict[str, Any]:
    """Return the model-extraction risk verdict for one client."""
    return get_extraction_detector().assess(client_id).to_dict()


@router.get("/extraction")
async def extraction_report() -> dict[str, Any]:
    """Return model-extraction verdicts for every observed client."""
    return get_extraction_detector().report()


@router.post("/poisoning")
async def detect_poisoning(request: PoisoningRequest) -> dict[str, Any]:
    """Screen a training set for poisoned samples."""
    features = _as_matrix(request.features)
    labels = _check_lengths(features, request.labels)
    detector = PoisoningDetector(
        n_neighbors=request.n_neighbors,
        outlier_z_threshold=request.outlier_z_threshold,
    )
    return detector.detect(features, labels).to_dict()


@router.post("/scan")
async def security_scan(request: SecurityScanRequest) -> dict[str, Any]:
    """Run the full security suite and return a scored report."""
    model = _resolve_model(request.model_name)

    eval_x = eval_y = train_x = train_y = None
    if request.eval_features is not None and request.eval_labels is not None:
        eval_x = _as_matrix(request.eval_features)
        eval_y = _check_lengths(eval_x, request.eval_labels)
    if request.train_features is not None and request.train_labels is not None:
        train_x = _as_matrix(request.train_features)
        train_y = _check_lengths(train_x, request.train_labels)

    pipeline = SecurityTestPipeline(
        request.model_name,
        model,
        attack_config=request.config.to_config(),
        extraction_detector=get_extraction_detector(),
    )
    result = pipeline.run(x_eval=eval_x, y_eval=eval_y, x_train=train_x, y_train=train_y)
    payload = result.to_dict()
    if request.include_markdown:
        payload["markdown"] = result.to_markdown()
    return payload
