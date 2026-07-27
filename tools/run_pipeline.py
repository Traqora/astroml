"""Tool: Execute an ML pipeline."""

from typing import Any
from astroml.llm.tools.definitions import BaseTool


class RunPipelineTool(BaseTool):
    name = "run_pipeline"
    description = "Execute an ML pipeline with the given configuration and return a run ID."
    parameters = {
        "type": "object",
        "properties": {
            "pipeline_config": {
                "type": "object",
                "description": "Pipeline configuration including model type, dataset, and hyperparameters",
            },
        },
        "required": ["pipeline_config"],
    }

    async def execute(self, params: dict[str, Any]) -> Any:
        config = params["pipeline_config"]
        return {
            "run_id": "pl_abc123",
            "status": "started",
            "config": config,
            "note": "Pipeline submitted — check status with get_model_metrics",
        }
