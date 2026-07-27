"""Agent router — LLM-powered agent task execution endpoints.

Resolves #457: Run autonomous LLM agent tasks with tool-use simulation,
task management, and result retrieval.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.services.llm import LLMService
from api.routers.llm import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm/agents", tags=["llm", "agents"])

# In-memory task store (replace with DB in production)
_task_store: dict[str, dict[str, Any]] = {}


class AgentTask(BaseModel):
    task: str = Field(..., min_length=1, description="The agent's goal or task description")
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names the agent may use",
    )
    model: str = Field("gpt-4-turbo")
    max_steps: int = Field(5, ge=1, le=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskResponse(BaseModel):
    task_id: str
    status: str = Field(..., pattern="^(pending|running|completed|failed)$")
    task: str
    result: str | None = None
    steps_taken: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/run",
    response_model=AgentTaskResponse,
    summary="Run an LLM agent task",
    operation_id="llm_agent_run",
)
async def run_agent(
    body: AgentTask,
    request: Request,
    service: LLMService = Depends(get_llm_service),
) -> AgentTaskResponse:
    """Execute an LLM agent task synchronously (short tasks).

    For long-running tasks, use POST /agents/submit for async execution.
    """
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    user_id = getattr(request.state, "user_id", None)

    tools_ctx = f"\nAvailable tools: {', '.join(body.tools)}" if body.tools else ""
    prompt = (
        f"You are an autonomous agent. Complete the following task step by step.\n"
        f"{tools_ctx}\n\nTask: {body.task}\n\n"
        f"Respond with your reasoning and final answer."
    )

    try:
        result = await service.generate(
            prompt=prompt,
            model=body.model,
            user_id=user_id,
            metadata={"task_id": task_id, **body.metadata},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid request") from exc

    task_record = AgentTaskResponse(
        task_id=task_id,
        status="completed",
        task=body.task,
        result=result["content"],
        steps_taken=1,
        cost=result["cost"],
        latency_ms=result["latency_ms"],
        metadata=body.metadata,
    )
    _task_store[task_id] = task_record.model_dump()
    return task_record


@router.get(
    "/{task_id}",
    response_model=AgentTaskResponse,
    summary="Get agent task status and result",
    operation_id="llm_agent_get_task",
)
async def get_agent_task(task_id: str) -> AgentTaskResponse:
    """Retrieve the status and result of a previously submitted agent task."""
    record = _task_store.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return AgentTaskResponse(**record)


@router.get(
    "/",
    summary="List agent tasks",
    operation_id="llm_agent_list_tasks",
)
async def list_agent_tasks(
    request: Request,
    limit: int = 20,
) -> dict[str, Any]:
    """List recent agent tasks."""
    tasks = list(reversed(list(_task_store.values())))[:limit]
    return {"tasks": tasks, "total": len(_task_store)}
