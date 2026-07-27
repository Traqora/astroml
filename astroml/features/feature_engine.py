"""Feature computation engine for the Feature Store.

This module provides the core computation engine that orchestrates feature
calculation using existing feature modules and manages the computation pipeline.
"""

from __future__ import annotations
from ..cache import cached_feature


import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Protocol,
    runtime_checkable,
)
from enum import Enum
import concurrent.futures
import threading
from contextlib import contextmanager

import pandas as pd
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)


class ComputationStatus(Enum):
    """Status of feature computation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeatureDependencyType(Enum):
    """Types of feature dependencies."""
    DATA = "data"  # Depends on raw data
    FEATURE = "feature"  # Depends on another feature
    EXTERNAL = "external"  # Depends on external data source


@dataclass
class FeatureDependency:
    """Definition of a feature dependency.
    
    Attributes:
        name: Dependency name
        dependency_type: Type of dependency
        parameters: Dependency parameters
        required: Whether this dependency is required
    """
    
    name: str
    dependency_type: FeatureDependencyType
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class ComputationTask:
    """A feature computation task.
    
    Attributes:
        task_id: Unique task identifier
        feature_name: Feature name to compute
        data: Input data
        parameters: Computation parameters
        dependencies: List of dependencies
        status: Computation status
        created_at: Task creation time
        started_at: Task start time
        completed_at: Task completion time
        error: Error information if failed
        result: Computation result
    """
    
    task_id: str
    feature_name: str
    data: pd.DataFrame
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[FeatureDependency] = field(default_factory=list)
    status: ComputationStatus = ComputationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[pd.DataFrame] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@runtime_checkable
class FeatureComputer(Protocol):
    """Protocol for feature computation functions."""
    
    def __call__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute features from input data.
        
        Args:
            data: Input DataFrame
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with computed features indexed by entity
        """
        ...


class BaseFeatureComputer(ABC):
    """Base class for feature computers with common functionality."""
    
    def __init__(self, name: str):
        """Initialize feature computer.
        
        Args:
            name: Feature computer name
        """
        self.name = name
        self._dependencies: List[FeatureDependency] = []
        self._parameters: Dict[str, Any] = {}
    
    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Get feature dependencies."""
        return self._dependencies.copy()
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get feature parameters."""
        return self._parameters.copy()
    
    def add_dependency(
        self,
        name: str,
        dependency_type: FeatureDependencyType,
        parameters: Optional[Dict[str, Any]] = None,
        required: bool = True,
    ) -> None:
        """Add a dependency.
        
        Args:
            name: Dependency name
            dependency_type: Type of dependency
            parameters: Dependency parameters
            required: Whether dependency is required
        """
        dependency = FeatureDependency(
            name=name,
            dependency_type=dependency_type,
            parameters=parameters or {},
            required=required,
        )
        self._dependencies.append(dependency)
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self._parameters[name] = value
    
    @abstractmethod
    @cached_feature(ttl_seconds=900)
    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute the feature.
        
        Args:
            data: Input data
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with computed features
        """
        pass
    
    def validate_input(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
    ) -> None:
        """Validate input data.
        
        Args:
            data: Input data
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            
        Raises:
            ValueError: If validation fails
        """
        required_cols = [entity_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if data[entity_col].isna().any():
            raise ValueError(f"Entity column '{entity_col}' contains null values")
        
        if data[timestamp_col].isna().any():
            raise ValueError(f"Timestamp column '{timestamp_col}' contains null values")


class FrequencyFeatureComputer(BaseFeatureComputer):
    """Computer for frequency-based features."""
    
    def __init__(self):
        super().__init__("frequency_features")
        
        # Add data dependencies
        self.add_dependency(
            "transaction_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "timestamp", "amount"]},
        )
    
    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute frequency features."""
        self.validate_input(data, entity_col, timestamp_col)
        
        try:
            from astroml.features.frequency import (
                compute_daily_transaction_counts,
                compute_burstiness,
            )
            
            # Compute daily transaction counts
            daily_counts = compute_daily_transaction_counts(
                data,
                entity_col=entity_col,
                timestamp_col=timestamp_col,
                **kwargs
            )
            
            # Compute burstiness
            burstiness = compute_burstiness(
                data,
                entity_col=entity_col,
                timestamp_col=timestamp_col,
                **kwargs
            )
            
            # Combine results
            result = pd.concat([daily_counts, burstiness], axis=1)
            result.columns = ["daily_transaction_count", "burstiness"]
            
            return result
            
        except ImportError as e:
            logger.error(f"Could not import frequency module: {e}")
            raise


class StructuralFeatureComputer(BaseFeatureComputer):
    """Computer for structural graph features."""
    
    def __init__(self):
        super().__init__("structural_features")
        
        # Add data dependencies
        self.add_dependency(
            "edge_data",
            FeatureDependencyType.DATA,
            {"columns": ["src", "dst", "amount", "timestamp"]},
        )
    
    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute structural features."""
        self.validate_input(data, entity_col, timestamp_col)
        
        try:
            from astroml.features.structural_importance import (
                compute_degree_centrality,
                compute_betweenness_centrality,
                compute_pagerank,
            )
            
            # Convert data to edge format
            edges = data.to_dict('records')
            
            # Compute centrality measures
            degree_centrality = compute_degree_centrality(edges, **kwargs)
            betweenness_centrality = compute_betweenness_centrality(edges, **kwargs)
            pagerank = compute_pagerank(edges, **kwargs)
            
            # Combine results
            result = pd.DataFrame({
                "degree_centrality": degree_centrality,
                "betweenness_centrality": betweenness_centrality,
                "pagerank": pagerank,
            })
            
            return result
            
        except ImportError as e:
            logger.error(f"Could not import structural importance module: {e}")
            raise


class NodeFeatureComputer(BaseFeatureComputer):
    """Computer for basic node features."""
    
    def __init__(self):
        super().__init__("node_features")
        
        # Add data dependencies
        self.add_dependency(
            "edge_data",
            FeatureDependencyType.DATA,
            {"columns": ["src", "dst", "amount", "timestamp"]},
        )
    
    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute node features."""
        self.validate_input(data, entity_col, timestamp_col)
        
        try:
            from astroml.features.node_features import compute_node_features
            
            # Convert data to edge format
            edges = data.to_dict('records')
            
            # Compute node features
            result = compute_node_features(edges, **kwargs)
            
            return result
            
        except ImportError as e:
            logger.error(f"Could not import node features module: {e}")
            raise


class AssetFeatureComputer(BaseFeatureComputer):
    """Computer for asset-related features."""
    
    def __init__(self):
        super().__init__("asset_features")
        
        # Add data dependencies
        self.add_dependency(
            "transaction_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "asset", "amount", "timestamp"]},
        )
    
    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute asset features."""
        self.validate_input(data, entity_col, timestamp_col)
        
        try:
            from astroml.features.asset_diversity import compute_asset_diversity
            
            # Compute asset diversity
            result = compute_asset_diversity(data, **kwargs)
            
            return result
            
        except ImportError as e:
            logger.error(f"Could not import asset diversity module: {e}")
            raise


class ComputationEngine:
    """Feature computation engine.
    
    Orchestrates feature computation with support for parallel processing,
    dependency resolution, and error handling.
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize computation engine.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self._computers: Dict[str, BaseFeatureComputer] = {}
        self._task_queue: List[ComputationTask] = []
        self._running_tasks: Dict[str, ComputationTask] = {}
        self._completed_tasks: Dict[str, ComputationTask] = {}
        self._lock = threading.Lock()
        self._register_builtin_computers()
    
    def _register_builtin_computers(self) -> None:
        """Register built-in feature computers."""
        self.register_computer(FrequencyFeatureComputer())
        self.register_computer(StructuralFeatureComputer())
        self.register_computer(NodeFeatureComputer())
        self.register_computer(AssetFeatureComputer())
        
        logger.info("Registered built-in feature computers")
    
    def register_computer(self, computer: BaseFeatureComputer) -> None:
        """Register a feature computer.
        
        Args:
            computer: Feature computer to register
        """
        self._computers[computer.name] = computer
        logger.info(f"Registered feature computer: {computer.name}")
    
    def get_computer(self, name: str) -> Optional[BaseFeatureComputer]:
        """Get a registered computer.
        
        Args:
            name: Computer name
            
        Returns:
            Computer if found, None otherwise
        """
        return self._computers.get(name)
    
    def list_computers(self) -> List[str]:
        """List all registered computers."""
        return list(self._computers.keys())
    
    def create_task(
        self,
        feature_name: str,
        data: pd.DataFrame,
        computer_name: str,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        **kwargs: Any,
    ) -> ComputationTask:
        """Create a computation task.
        
        Args:
            feature_name: Feature name
            data: Input data
            computer_name: Computer to use
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters
            
        Returns:
            Created computation task
        """
        import uuid
        
        task = ComputationTask(
            task_id=str(uuid.uuid4()),
            feature_name=feature_name,
            data=data,
            parameters={
                "computer_name": computer_name,
                "entity_col": entity_col,
                "timestamp_col": timestamp_col,
                **kwargs
            }
        )
        
        # Add computer dependencies
        computer = self.get_computer(computer_name)
        if computer:
            task.dependencies = computer.dependencies
        
        return task
    
    def submit_task(self, task: ComputationTask) -> None:
        """Submit a task for computation.
        
        Args:
            task: Task to submit
        """
        with self._lock:
            self._task_queue.append(task)
        
        logger.info(f"Submitted task {task.task_id} for feature {task.feature_name}")
    
    def submit_tasks(self, tasks: List[ComputationTask]) -> None:
        """Submit multiple tasks for computation.
        
        Args:
            tasks: Tasks to submit
        """
        with self._lock:
            self._task_queue.extend(tasks)
        
        logger.info(f"Submitted {len(tasks)} tasks for computation")
    
    def _execute_task(self, task: ComputationTask) -> None:
        """Execute a single task.
        
        Args:
            task: Task to execute
        """
        try:
            task.status = ComputationStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Get computer
            computer_name = task.parameters.get("computer_name")
            computer = self.get_computer(computer_name)
            
            if not computer:
                raise ValueError(f"Computer '{computer_name}' not found")
            
            # Execute computation
            entity_col = task.parameters.get("entity_col", "entity_id")
            timestamp_col = task.parameters.get("timestamp_col", "timestamp")
            computation_kwargs = {
                k: v for k, v in task.parameters.items()
                if k not in ["computer_name", "entity_col", "timestamp_col"]
            }
            
            result = computer.compute(
                task.data,
                entity_col=entity_col,
                timestamp_col=timestamp_col,
                **computation_kwargs
            )
            
            task.result = result
            task.status = ComputationStatus.COMPLETED
            
            logger.info(f"Completed task {task.task_id} for feature {task.feature_name}")
            
        except Exception as e:
            task.error = str(e)
            task.status = ComputationStatus.FAILED
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            task.completed_at = datetime.utcnow()
    
    def run_tasks(self, parallel: bool = True) -> Dict[str, ComputationTask]:
        """Run all submitted tasks.
        
        Args:
            parallel: Whether to run tasks in parallel
            
        Returns:
            Dictionary of completed tasks
        """
        with self._lock:
            tasks = self._task_queue.copy()
            self._task_queue.clear()
        
        if not tasks:
            logger.info("No tasks to run")
            return {}
        
        logger.info(f"Running {len(tasks)} tasks (parallel={parallel})")
        
        if parallel and len(tasks) > 1:
            # Run tasks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._execute_task, task): task for task in tasks}
                
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()  # Wait for completion
                        self._completed_tasks[task.task_id] = task
                    except Exception as e:
                        logger.error(f"Task execution error: {e}")
                        self._completed_tasks[task.task_id] = task
        else:
            # Run tasks sequentially
            for task in tasks:
                self._execute_task(task)
                self._completed_tasks[task.task_id] = task
        
        logger.info(f"Completed {len(self._completed_tasks)} tasks")
        return self._completed_tasks.copy()
    
    def get_task(self, task_id: str) -> Optional[ComputationTask]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        return self._completed_tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[ComputationStatus]:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status if found, None otherwise
        """
        task = self.get_task(task_id)
        return task.status if task else None
    
    def clear_completed_tasks(self) -> None:
        """Clear completed tasks."""
        with self._lock:
            self._completed_tasks.clear()
        
        logger.info("Cleared completed tasks")
    
    @contextmanager
    def computation_context(self):
        """Context manager for computation operations."""
        try:
            yield self
        finally:
            self.clear_completed_tasks()
    
    def compute_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        computer_name: str,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute a single feature.
        
        Args:
            feature_name: Feature name
            data: Input data
            computer_name: Computer to use
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters
            
        Returns:
            Computed feature values
        """
        task = self.create_task(
            feature_name=feature_name,
            data=data,
            computer_name=computer_name,
            entity_col=entity_col,
            timestamp_col=timestamp_col,
            **kwargs
        )
        
        from astroml.observability.metrics import FEATURE_COMPUTE_TIME

        self.submit_task(task)
        # Feature computation time (issue #567).
        with FEATURE_COMPUTE_TIME.labels(feature_name).time():
            completed_tasks = self.run_tasks(parallel=False)

        if task.task_id not in completed_tasks:
            raise RuntimeError(f"Task {task.task_id} not found in completed tasks")
        
        completed_task = completed_tasks[task.task_id]
        
        if completed_task.status != ComputationStatus.COMPLETED:
            raise RuntimeError(f"Task failed: {completed_task.error}")
        
        return completed_task.result
    
    def compute_features_batch(
        self,
        feature_configs: List[Dict[str, Any]],
        data: pd.DataFrame,
        parallel: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Compute multiple features in batch.
        
        Args:
            feature_configs: List of feature configuration dictionaries
            data: Input data
            parallel: Whether to run in parallel
            
        Returns:
            Dictionary of feature names to computed values
        """
        tasks = []
        
        for config in feature_configs:
            task = self.create_task(
                feature_name=config["name"],
                data=data,
                computer_name=config["computer"],
                entity_col=config.get("entity_col", "entity_id"),
                timestamp_col=config.get("timestamp_col", "timestamp"),
                **config.get("parameters", {})
            )
            tasks.append(task)
        
        self.submit_tasks(tasks)
        completed_tasks = self.run_tasks(parallel=parallel)
        
        results = {}
        for task in tasks:
            if task.task_id in completed_tasks:
                completed_task = completed_tasks[task.task_id]
                if completed_task.status == ComputationStatus.COMPLETED:
                    results[task.feature_name] = completed_task.result
                else:
                    logger.error(f"Task {task.task_id} failed: {completed_task.error}")
        
        return results


# Decorator for feature computation functions

def feature_computer(
    name: str,
    dependencies: Optional[List[Dict[str, Any]]] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """Decorator to create feature computers from functions.
    
    Args:
        name: Feature computer name
        dependencies: List of dependency specifications
        parameters: Default parameters
    """
    def decorator(func: Callable) -> BaseFeatureComputer:
        class DecoratedComputer(BaseFeatureComputer):
            def __init__(self):
                super().__init__(name)
                
                # Add dependencies
                if dependencies:
                    for dep_config in dependencies:
                        self.add_dependency(
                            dep_config["name"],
                            FeatureDependencyType(dep_config["type"]),
                            dep_config.get("parameters", {}),
                            dep_config.get("required", True)
                        )
                
                # Set parameters
                if parameters:
                    for param_name, param_value in parameters.items():
                        self.set_parameter(param_name, param_value)
            
            def compute(
                self,
                data: pd.DataFrame,
                entity_col: str,
                timestamp_col: str,
                **kwargs: Any,
            ) -> pd.DataFrame:
                return func(data, entity_col, timestamp_col, **kwargs)
        
        return DecoratedComputer()
    
    return decorator


# Convenience functions

def create_computation_engine(max_workers: int = 4) -> ComputationEngine:
    """Create a computation engine instance.
    
    Args:
        max_workers: Maximum number of parallel workers
        
    Returns:
        Computation engine instance
    """
    return ComputationEngine(max_workers=max_workers)


def compute_feature(
    feature_name: str,
    data: pd.DataFrame,
    computer_name: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute a single feature using the default engine.
    
    Args:
        feature_name: Feature name
        data: Input data
        computer_name: Computer to use
        **kwargs: Additional parameters
        
    Returns:
        Computed feature values
    """
    engine = create_computation_engine()
    return engine.compute_feature(feature_name, data, computer_name, **kwargs)
