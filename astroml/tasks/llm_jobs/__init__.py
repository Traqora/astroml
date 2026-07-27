"""LLM backfill job type handlers."""

from .embedding_job import EmbeddingJobHandler
from .explanation_job import ExplanationJobHandler
from .label_job import LabelJobHandler
from .report_job import ReportJobHandler

JOB_HANDLERS = {
    "embedding": EmbeddingJobHandler(),
    "explanation": ExplanationJobHandler(),
    "label": LabelJobHandler(),
    "report": ReportJobHandler(),
}


def get_job_handler(job_type: str):
    handler = JOB_HANDLERS.get(job_type)
    if handler is None:
        raise ValueError(f"Unknown job type: {job_type}")
    return handler


__all__ = [
    "JOB_HANDLERS",
    "get_job_handler",
    "EmbeddingJobHandler",
    "ExplanationJobHandler",
    "LabelJobHandler",
    "ReportJobHandler",
]
