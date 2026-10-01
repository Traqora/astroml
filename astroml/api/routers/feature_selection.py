from astroml.utils.exceptions import AstroMLError
"""Feature selection API router for AstroML.

Provides REST endpoints for filter, wrapper, embedded, hybrid and pipelined
feature selection, plus held-out evaluation of a stored selection.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from astroml.preprocessing.feature_selection.embedded import EmbeddedSelector
from astroml.preprocessing.feature_selection.filter import FilterSelector, SelectionResult
from astroml.preprocessing.feature_selection.hybrid import (
    FeatureSelectionPipeline,
    HybridSelector,
)
from astroml.preprocessing.feature_selection.wrapper import WrapperSelector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["feature-selection"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class FeatureMatrixRequest(BaseModel):
    """Request body for feature selection on a matrix."""

    data: list[list[float]] = Field(..., description="Feature matrix rows")
    target: list[float] = Field(..., description="Target values")
    feature_names: list[str] | None = Field(default=None)
    method: str = Field(
        default="mutual_info",
        description="Selection method or strategy name",
    )
    k: int | None = Field(default=None, description="Number of features to select")
    threshold: float = Field(default=0.0, description="Score threshold")

    model_config = ConfigDict(extra="forbid")


class FeatureSetRequest(BaseModel):
    """Request body for hybrid selection with multiple strategies."""

    data: list[list[float]] = Field(...)
    target: list[float] = Field(...)
    feature_names: list[str] | None = Field(default=None)
    strategy: str = Field(default="vote", description="vote|rank_aggregation|intersection|union")
    methods: list[str] = Field(
        default=["mutual_info", "correlation", "variance"],
        description="Filter methods to ensemble",
    )
    k: int | None = Field(default=None)
    min_votes: int = Field(default=2)

    model_config = ConfigDict(extra="forbid")


class PipelineStepConfig(BaseModel):
    """One stage of a feature-selection pipeline."""

    type: Literal["filter", "wrapper", "embedded"] = Field(..., description="Selector kind")
    method: str = Field(..., description="Method name within that selector kind")
    k: int | None = Field(default=None, description="Features to keep (filter/embedded)")
    threshold: float = Field(default=0.0, description="Score threshold")
    estimator: str = Field(
        default="decision_tree",
        description="Wrapper estimator name (wrapper steps only)",
    )
    task: Literal["classification", "regression"] = Field(
        default="classification",
        description="Wrapper estimator kind (wrapper steps only)",
    )

    model_config = ConfigDict(extra="forbid")


class WrapperRequest(BaseModel):
    """Request body for wrapper selection (RFE, forward, backward)."""

    data: list[list[float]] = Field(...)
    target: list[float] = Field(...)
    feature_names: list[str] | None = Field(default=None)
    method: str = Field(default="rfe", description="rfe|forward|backward")
    estimator: str = Field(default="decision_tree", description="Estimator name")
    task: Literal["classification", "regression"] = Field(default="classification")
    n_features_to_select: int = Field(default=10, ge=1)
    step: int = Field(default=1, ge=1, description="Features added/removed per iteration")
    scoring: str | None = Field(default=None, description="sklearn scoring name")
    cv: int | None = Field(default=None, ge=2, le=10, description="Folds for forward/backward")

    model_config = ConfigDict(extra="forbid")


class PipelineRequest(BaseModel):
    """Request body for pipeline selection."""

    data: list[list[float]] = Field(...)
    target: list[float] = Field(...)
    feature_names: list[str] | None = Field(default=None)
    steps: list[PipelineStepConfig] = Field(
        ...,
        description=(
            "Pipeline steps executed in order, e.g. "
            "[{type: filter, method: mutual_info, k: 50}, "
            "{type: embedded, method: tree, k: 20}]"
        ),
    )

    model_config = ConfigDict(extra="forbid")


class SelectionResponse(BaseModel):
    """Selected features for one ``POST /feature-selection/*`` call."""

    selection_id: str
    method: str
    num_features_selected: int
    num_features_total: int
    selected_indices: list[int]
    scores: list[float]
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EvaluationResponse(BaseModel):
    """Scores for a stored selection, returned by the evaluate endpoint."""

    selection_id: str
    train_score: float | None = None
    test_score: float | None = None
    cv_score: float | None = None
    cv_std: float | None = None
    run_time_ms: float | None = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# In-memory selection store
# ---------------------------------------------------------------------------


@dataclass
class _StoredSelection:
    """A fitted selector plus the data it was fitted on.

    The matrix is kept so ``POST /feature-selection/evaluate/{id}`` can score
    the reduced feature set without the client re-posting it.
    """

    selector: Any
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    result: SelectionResult


# Selections live for the process lifetime, so the store is LRU-capped: left
# unbounded it would grow with every request against a long-running server.
MAX_STORED_SELECTIONS = 128
_selections: OrderedDict[str, _StoredSelection] = OrderedDict()


def _store_selection(
    selector: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    result: SelectionResult,
) -> str:
    """Remember a fitted selection and return its id.

    Args:
        selector: Fitted selector exposing ``transform``.
        X: Feature matrix the selector was fitted on.
        y: Target vector the selector was fitted on.
        result: Structured selection result to echo back.

    Returns:
        Opaque selection id usable with the evaluate endpoint.
    """
    selection_id = uuid.uuid4().hex[:12]
    _selections[selection_id] = _StoredSelection(selector=selector, X=X, y=y, result=result)
    _selections.move_to_end(selection_id)
    while len(_selections) > MAX_STORED_SELECTIONS:
        _selections.popitem(last=False)
    return selection_id


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_matrix(
    data: list[list[float]],
    target: list[float],
    feature_names: list[str] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert request payload into aligned numpy arrays.

    Rejects malformed payloads as ``422`` rather than letting a ragged array or
    a NaN reach numpy/sklearn, where it surfaces as an opaque ``500``.

    Args:
        data: Feature matrix rows.
        target: Target values, one per row.
        feature_names: Optional column names.

    Returns:
        ``(X, y)`` as float64 arrays.

    Raises:
        HTTPException: 422 when the payload is empty, ragged, misaligned,
            non-numeric, or contains NaN/inf.
    """
    if not data or not data[0]:
        raise HTTPException(status_code=422, detail="data must be a non-empty matrix")
    widths = {len(row) for row in data}
    if len(widths) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"data rows must all have the same width, got {sorted(widths)}",
        )
    if len(target) != len(data):
        raise HTTPException(
            status_code=422,
            detail=f"target has {len(target)} values but data has {len(data)} rows",
        )
    if feature_names is not None and len(feature_names) != len(data[0]):
        raise HTTPException(
            status_code=422,
            detail=(
                f"feature_names has {len(feature_names)} entries but data has "
                f"{len(data[0])} columns"
            ),
        )
    try:
        X = np.asarray(data, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"data and target must be numeric: {exc}"
        ) from exc
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise HTTPException(
            status_code=422, detail="data and target must be finite (no NaN or infinity)"
        )
    return X, y


def _selection_response(
    selection_id: str,
    method: str,
    result: SelectionResult,
) -> SelectionResponse:
    """Build the API response for a fitted selection.

    Args:
        selection_id: Id returned by :func:`_store_selection`.
        method: Human-readable selector label.
        result: Structured selection result.

    Returns:
        Response payload.
    """
    return SelectionResponse(
        selection_id=selection_id,
        method=method,
        num_features_selected=result.num_features_selected,
        num_features_total=result.num_features_total,
        selected_indices=result.selected_indices,
        scores=[round(score, 6) for score in result.scores],
        feature_names=result.feature_names,
        metadata=result.metadata,
    )


# Wrapper estimators are named, not passed as objects: a JSON body cannot carry
# a fitted sklearn estimator.  ``name -> (classifier, regressor)`` import paths.
WRAPPER_ESTIMATORS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "decision_tree": (
        ("sklearn.tree", "DecisionTreeClassifier"),
        ("sklearn.tree", "DecisionTreeRegressor"),
    ),
    "random_forest": (
        ("sklearn.ensemble", "RandomForestClassifier"),
        ("sklearn.ensemble", "RandomForestRegressor"),
    ),
    "extra_trees": (
        ("sklearn.ensemble", "ExtraTreesClassifier"),
        ("sklearn.ensemble", "ExtraTreesRegressor"),
    ),
    "gradient_boosting": (
        ("sklearn.ensemble", "GradientBoostingClassifier"),
        ("sklearn.ensemble", "GradientBoostingRegressor"),
    ),
}


def _build_wrapper_estimator(estimator: str, task: str) -> Any:
    """Instantiate a named estimator for wrapper selection.

    Args:
        estimator: Key of :data:`WRAPPER_ESTIMATORS`.
        task: ``"classification"`` or ``"regression"``.

    Returns:
        An unfitted, seed-fixed scikit-learn estimator.

    Raises:
        HTTPException: 422 for an unknown name, 503 when scikit-learn is absent.
    """
    entry = WRAPPER_ESTIMATORS.get(estimator)
    if entry is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported wrapper estimator '{estimator}'. "
                f"Choose from: {', '.join(sorted(WRAPPER_ESTIMATORS))}"
            ),
        )
    module_path, class_name = entry[0] if task == "classification" else entry[1]
    try:
        estimator_cls = getattr(import_module(module_path), class_name)
    except ImportError as exc:  # pragma: no cover - scikit-learn is a hard dep
        raise HTTPException(
            status_code=503,
            detail=f"scikit-learn is required for wrapper selection: {exc}",
        ) from exc
    return estimator_cls(random_state=42)


# ---------------------------------------------------------------------------
# Filter endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/filter",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run filter-based feature selection",
)
async def filter_selection(request: FeatureMatrixRequest) -> SelectionResponse:
    """Score every feature against the target and keep the best ones.

    Args:
        request: Matrix, target, and filter configuration.

    Returns:
        The fitted selection and its per-feature scores.
    """
    try:
        X, y = _as_matrix(request.data, request.target, request.feature_names)

        selector = FilterSelector(
            method=request.method,
            k=request.k,
            threshold=request.threshold,
        )
        selector.fit(X, y, request.feature_names)
        result = selector.get_selection_result()

        selection_id = _store_selection(selector, X, y, result)
        return _selection_response(selection_id, f"filter-{request.method}", result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in filter selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Wrapper endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/wrapper",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run wrapper-based feature selection",
)
async def wrapper_selection(request: WrapperRequest) -> SelectionResponse:
    """Search feature subsets by model performance (RFE, forward, backward).

    Wrapper search refits an estimator per candidate subset, so cost grows with
    ``n_features_to_select`` and the number of columns; it is deliberately not
    cross-validated by default.

    Args:
        request: Matrix, target, method, and estimator configuration.

    Returns:
        The fitted selection and its per-feature scores.
    """
    try:
        X, y = _as_matrix(request.data, request.target, request.feature_names)
        model = _build_wrapper_estimator(request.estimator, request.task)

        selector = WrapperSelector(
            estimator=model,
            method=request.method,
            n_features_to_select=min(request.n_features_to_select, X.shape[1]),
            step=request.step,
            scoring=request.scoring,
            cv=request.cv,
        )
        selector.fit(X, y, request.feature_names)
        result = selector.get_selection_result()

        selection_id = _store_selection(selector, X, y, result)
        return _selection_response(selection_id, f"wrapper-{request.method}", result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in wrapper selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Embedded endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/embedded",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run embedded feature selection",
)
async def embedded_selection(request: FeatureMatrixRequest) -> SelectionResponse:
    """Select features from a model that performs selection while training.

    Args:
        request: Matrix, target, and embedded configuration.

    Returns:
        The fitted selection and its per-feature importances.
    """
    try:
        X, y = _as_matrix(request.data, request.target, request.feature_names)

        selector = EmbeddedSelector(
            method=request.method if request.method in ("lasso", "tree", "elasticnet") else "tree",
            threshold=request.threshold,
            k=request.k,
        )
        selector.fit(X, y, request.feature_names)
        result = selector.get_selection_result()

        selection_id = _store_selection(selector, X, y, result)
        return _selection_response(selection_id, f"embedded-{request.method}", result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in embedded selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Hybrid / ensemble endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/hybrid",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run hybrid feature selection (ensemble of filters)",
)
async def hybrid_selection(request: FeatureSetRequest) -> SelectionResponse:
    """Combine several filters into one selection by vote, union or ranking.

    Args:
        request: Matrix, target, member methods, and combination strategy.

    Returns:
        The fitted selection and its aggregated scores.
    """
    try:
        X, y = _as_matrix(request.data, request.target, request.feature_names)

        # ``list`` is invariant, so the element type has to match HybridSelector's
        # own annotation rather than the narrower FilterSelector.
        selectors: list[tuple[str, FilterSelector | WrapperSelector | EmbeddedSelector]] = [
            (method, FilterSelector(method=method, k=request.k)) for method in request.methods
        ]

        hybrid = HybridSelector(
            selectors=selectors,
            strategy=request.strategy,
            min_votes=request.min_votes,
            k=request.k,
        )
        hybrid.fit(X, y, request.feature_names)
        result = hybrid.get_selection_result()

        selection_id = _store_selection(hybrid, X, y, result)
        return _selection_response(selection_id, f"hybrid-{request.strategy}", result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in hybrid selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Pipeline endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/pipeline",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run a chained feature selection pipeline",
)
async def pipeline_selection(request: PipelineRequest) -> SelectionResponse:
    """Run several selectors in sequence, each narrowing the previous output.

    Args:
        request: Matrix, target, and ordered pipeline steps.

    Returns:
        The final selection.  ``selected_indices`` and ``feature_names`` are
        mapped back to the *original* columns; ``metadata["steps"]`` carries the
        per-step results.
    """
    try:
        X, y = _as_matrix(request.data, request.target, request.feature_names)
        if not request.steps:
            raise HTTPException(status_code=422, detail="steps must not be empty")

        pipeline = FeatureSelectionPipeline(
            [(step.type, _build_step_selector(step)) for step in request.steps]
        )
        pipeline.fit(X, y, request.feature_names)
        results = pipeline.get_results()
        if not results:  # pragma: no cover - guarded by the empty-steps check
            raise HTTPException(status_code=500, detail="pipeline produced no results")

        final = results[-1]
        absolute_indices = _map_to_original_indices(results, X.shape[1])
        result = SelectionResult(
            selector_name=final.selector_name,
            num_features_selected=len(absolute_indices),
            num_features_total=X.shape[1],
            selected_indices=absolute_indices,
            scores=final.scores,
            feature_names=(
                [request.feature_names[i] for i in absolute_indices]
                if request.feature_names
                else None
            ),
            metadata={
                **final.metadata,
                "pipeline": pipeline.summary(),
                "steps": [
                    {
                        "type": step.type,
                        "method": step.method,
                        "num_features_selected": step_result.num_features_selected,
                        "num_features_total": step_result.num_features_total,
                    }
                    for step, step_result in zip(request.steps, results)
                ],
            },
        )

        selection_id = _store_selection(_PipelineSelector(pipeline), X, y, result)
        return _selection_response(
            selection_id,
            f"pipeline-{'-'.join(step.type for step in request.steps)}",
            result,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in pipeline selection")
        raise HTTPException(status_code=500, detail=str(e))


def _build_step_selector(step: PipelineStepConfig) -> Any:
    """Instantiate the selector for one pipeline step.

    Args:
        step: Step configuration.

    Returns:
        A configured, unfitted selector.

    Raises:
        HTTPException: 422 for an unknown step type.
    """
    if step.type == "filter":
        return FilterSelector(method=step.method, k=step.k, threshold=step.threshold)
    if step.type == "embedded":
        return EmbeddedSelector(method=step.method, k=step.k, threshold=step.threshold)
    if step.type == "wrapper":
        return WrapperSelector(
            estimator=_build_wrapper_estimator(step.estimator, step.task),
            method=step.method,
            n_features_to_select=step.k or 10,
        )
    raise HTTPException(  # pragma: no cover - Literal rejects this first
        status_code=422, detail=f"Unknown pipeline step type '{step.type}'"
    )


def _map_to_original_indices(
    results: list[SelectionResult],
    n_features_total: int,
) -> list[int]:
    """Compose per-step selections into indices over the original columns.

    Each step reports indices relative to its own (already reduced) input, so
    the chain has to be applied in order to say anything useful about the
    original matrix.

    Args:
        results: Per-step results, in execution order.
        n_features_total: Column count of the original matrix.

    Returns:
        Sorted indices into the original columns.
    """
    current = list(range(n_features_total))
    for result in results:
        current = [current[i] for i in result.selected_indices]
    return sorted(current)


class _PipelineSelector:
    """Adapter exposing a fitted pipeline through the selector interface."""

    def __init__(self, pipeline: FeatureSelectionPipeline) -> None:
        """Wrap a fitted pipeline.

        Args:
            pipeline: Pipeline already fitted on the stored matrix.
        """
        self._pipeline = pipeline

    def transform(
        self, X: NDArray[np.float64], y: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Reduce ``X`` through every pipeline step.

        Args:
            X: Feature matrix with the original column count.
            y: Ignored, accepted for selector compatibility.

        Returns:
            The reduced feature matrix.
        """
        return self._pipeline.transform(X)


# ---------------------------------------------------------------------------
# Evaluation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/evaluate/{selection_id}",
    response_model=EvaluationResponse,
    summary="Evaluate a feature selection result",
)
async def evaluate_selection(
    selection_id: str,
    cv: int = Query(default=5, ge=2, le=20),
) -> EvaluationResponse:
    """Score a stored selection against the data it was fitted on.

    A random forest is refit on the selected columns and reported three ways:
    cross-validated, on the training split, and on a held-out test split.  The
    estimator is chosen from the target itself so a continuous target is
    evaluated by regression rather than crashing the classifier.

    Args:
        selection_id: Id returned by a selection endpoint.
        cv: Number of cross-validation folds.

    Returns:
        Scores for the reduced feature set.

    Raises:
        HTTPException: 404 for an unknown id, 422 for too few rows to split,
            500 when scikit-learn is unavailable.
    """
    try:
        entry = _selections.get(selection_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Selection not found")

        X_selected = entry.selector.transform(entry.X)
        y = entry.y

        n_rows = X_selected.shape[0]
        if n_rows < 4:
            raise HTTPException(
                status_code=422,
                detail="need at least 4 rows to train and hold out a test split",
            )

        t0 = time.monotonic()
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import cross_val_score
        except ImportError as exc:  # pragma: no cover - scikit-learn is a hard dep
            raise HTTPException(
                status_code=500, detail=f"scikit-learn is required for evaluation: {exc}"
            ) from exc

        is_classification = len(np.unique(y)) <= 2
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        cv_scores = cross_val_score(model, X_selected, y, cv=min(cv, n_rows // 2))
        cv_score_mean = float(np.mean(cv_scores))
        cv_score_std = float(np.std(cv_scores))

        # Deterministic holdout: fit on two thirds, score the remaining third.
        order = np.random.default_rng(42).permutation(n_rows)
        cut = n_rows * 2 // 3
        train_idx, test_idx = order[:cut], order[cut:]
        model.fit(X_selected[train_idx], y[train_idx])
        train_score = float(model.score(X_selected[train_idx], y[train_idx]))
        test_score = float(model.score(X_selected[test_idx], y[test_idx]))

        runtime = (time.monotonic() - t0) * 1000.0

        return EvaluationResponse(
            selection_id=selection_id,
            train_score=round(train_score, 4),
            test_score=round(test_score, 4),
            cv_score=round(cv_score_mean, 4),
            cv_std=round(cv_score_std, 4),
            run_time_ms=round(runtime, 2),
        )
    except HTTPException:
        raise
    except AstroMLError as e:
        logger.exception("Error evaluating selection")
        raise HTTPException(status_code=500, detail=str(e))
