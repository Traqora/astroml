"""AstroML automated pipeline DAG definitions for Airflow and Dagster."""

from .deployment_pipeline import create_deployment_dag
from .evaluation_pipeline import create_evaluation_dag
from .training_pipeline import create_training_dag

__all__ = [
    "create_training_dag",
    "create_evaluation_dag",
    "create_deployment_dag",
]
