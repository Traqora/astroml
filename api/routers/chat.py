"""Chat router for issue #306.

Provides endpoints for:
- WebSocket chat connections
- Chat session management
- Agent management
- Offline message handling
- Chat transcripts
"""
from __future__ import annotations

import json
import uuid
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from astroml.chat import chat_service
from astroml.chat.models import (
    ChatSession,
    ChatMessage,
    Agent,
    AgentStatus,
    ChatStatus,
    MessageRole,
    OfflineMessage,
)
from astroml.chat.slack import SlackIntegration, SlackConfig

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


# ─── Request/Response Schemas ─────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    """Request schema for creating a chat session."""

    user_id: str = Field(..., description="User identifier")
    user_name: str = Field(..., description="User display name")
    user_email: Optional[str] = Field(None, description="User email")


class MessageRequest(BaseModel):
    """Request schema for sending a message."""

    session_id: str = Field(..., description="Session identifier")
    content: str = Field(..., description="Message content")
    sender_id: Optional[str] = Field(None, description="Sender identifier")
    sender_name: Optional[str] = Field(None, description="Sender name")


class AssignAgentRequest(BaseModel):
    """Request schema for assigning an agent."""

    session_id: str = Field(..., description="Session identifier")
    agent_id: str = Field(..., description="Agent identifier")


class RegisterAgentRequest(BaseModel):
    """Request schema for registering an agent."""

    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    email: str = Field(..., description="Agent email")
    slack_user_id: Optional[str] = Field(None, description="Slack user ID")
    max_concurrent_chats: int = Field(default=5, ge=1, le=20)


class SetAgentStatusRequest(BaseModel):
    """Request schema for setting agent status."""

    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Agent status: online, busy, offline, away")


class OfflineMessageRequest(BaseModel):
    """Request schema for offline messages."""

    user_name: str = Field(..., description="User name")
    user_email: str = Field(..., description="User email")
    message: str = Field(..., description="Message content")


# ─── WebSocket Chat Endpoint ─────────────────────────────────────────────

class ConnectionManager:
    """Manager for WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}  # session_id -> {connection_id: websocket}

    async def connect(self, websocket: WebSocket, session_id: str, connection_id: str):
        """Connect a WebSocket to a session."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}
        self.active_connections[session_id][connection_id] = websocket
        chat_service.add_connection(session_id, connection_id)
        logger.info(f"WebSocket connected: session={session_id}, connection={connection_id}")

    def disconnect(self, session_id: str, connection_id: str):
        """Disconnect a WebSocket."""
        if session_id in self.active_connections:
            self.active_connections[session_id].pop(connection_id, None)
            chat_service.remove_connection(session_id, connection_id)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected: session={session_id}, connection={connection_id}")

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast a message to all connections in a session."""
        if session_id in self.active_connections:
            for connection_id, websocket in self.active_connections[session_id].items():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to connection {connection_id}: {e}")
                    self.disconnect(session_id, connection_id)


manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id, connection_id)

    try:
        while True:
            data = await websocket.receive_json()

            # Handle incoming message
            message_type = data.get("type", "message")

            if message_type == "message":
                content = data.get("content", "")
                role = MessageRole.USER
                sender_id = data.get("sender_id")
                sender_name = data.get("sender_name")

                # Add message to session
                message = chat_service.add_message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    sender_id=sender_id,
                    sender_name=sender_name,
                )

                if message:
                    # Broadcast to all connections in session
                    await manager.broadcast_to_session(session_id, message.to_dict())

            elif message_type == "typing":
                # Broadcast typing indicator
                await manager.broadcast_to_session(session_id, {
                    "type": "typing",
                    "sender_id": data.get("sender_id"),
                    "is_typing": data.get("is_typing", False),
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id, connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id, connection_id)


# ─── REST API Endpoints ─────────────────────────────────────────────────

@router.post("/sessions", response_model=Dict[str, Any])
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    session = chat_service.create_session(
        user_id=request.user_id,
        user_name=request.user_name,
        user_email=request.user_email,
    )

    # Notify Slack if configured
    slack_config = SlackConfig.create_slack_config_from_env()
    if slack_config.webhook_url:
        slack = SlackIntegration(slack_config)
        slack.notify_new_chat(request.user_name, session.id)

    return session.to_dict()


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """Get a chat session by ID."""
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.post("/messages", response_model=Dict[str, Any])
async def send_message(request: MessageRequest):
    """Send a message to a chat session."""
    role = MessageRole.AGENT  # Default to agent for REST API
    message = chat_service.add_message(
        session_id=request.session_id,
        role=role,
        content=request.content,
        sender_id=request.sender_id,
        sender_name=request.sender_name,
    )

    if not message:
        raise HTTPException(status_code=404, detail="Session not found")

    # Broadcast via WebSocket
    await manager.broadcast_to_session(request.session_id, message.to_dict())

    return message.to_dict()


@router.post("/sessions/assign", response_model=Dict[str, Any])
async def assign_agent(request: AssignAgentRequest):
    """Assign an agent to a chat session."""
    success = chat_service.assign_agent(request.session_id, request.agent_id)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign agent")

    session = chat_service.get_session(request.session_id)

    # Notify Slack if configured
    if session and session.assigned_agent_name:
        slack_config = SlackConfig.create_slack_config_from_env()
        if slack_config.webhook_url:
            slack = SlackIntegration(slack_config)
            slack.notify_agent_assigned(session.assigned_agent_name, request.session_id)

    return {"success": True, "session_id": request.session_id, "agent_id": request.agent_id}


@router.post("/sessions/{session_id}/close")
async def close_session(session_id: str):
    """Close a chat session."""
    success = chat_service.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "session_id": session_id}


@router.post("/sessions/{session_id}/transfer")
async def transfer_session(session_id: str, new_agent_id: str):
    """Transfer a session to another agent."""
    success = chat_service.transfer_session(session_id, new_agent_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to transfer session")
    return {"success": True, "session_id": session_id, "new_agent_id": new_agent_id}


@router.get("/sessions/waiting", response_model=List[Dict[str, Any]])
async def get_waiting_sessions():
    """Get all sessions waiting for an agent."""
    sessions = chat_service.get_waiting_sessions()
    return [s.to_dict() for s in sessions]


@router.get("/agents", response_model=List[Dict[str, Any]])
async def get_agents():
    """Get all registered agents."""
    agents = list(chat_service.agents.values())
    return [a.to_dict() for a in agents]


@router.get("/agents/online", response_model=List[Dict[str, Any]])
async def get_online_agents():
    """Get all online agents."""
    agents = chat_service.get_online_agents()
    return [a.to_dict() for a in agents]


@router.post("/agents/register", response_model=Dict[str, Any])
async def register_agent(request: RegisterAgentRequest):
    """Register a new support agent."""
    agent = chat_service.register_agent(
        agent_id=request.agent_id,
        name=request.name,
        email=request.email,
        slack_user_id=request.slack_user_id,
        max_concurrent_chats=request.max_concurrent_chats,
    )
    return agent.to_dict()


@router.post("/agents/status")
async def set_agent_status(request: SetAgentStatusRequest):
    """Set agent status."""
    try:
        status = AgentStatus(request.status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid status")

    success = chat_service.set_agent_status(request.agent_id, status)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"success": True, "agent_id": request.agent_id, "status": request.status}


@router.get("/agents/{agent_id}/sessions", response_model=List[Dict[str, Any]])
async def get_agent_sessions(agent_id: str):
    """Get all sessions assigned to an agent."""
    sessions = chat_service.get_agent_sessions(agent_id)
    return [s.to_dict() for s in sessions]


@router.post("/offline-messages", response_model=Dict[str, Any])
async def create_offline_message(request: OfflineMessageRequest):
    """Create an offline message when no agents are available."""
    message = chat_service.create_offline_message(
        user_name=request.user_name,
        user_email=request.user_email,
        message=request.message,
    )

    # Notify Slack if configured
    slack_config = SlackConfig.create_slack_config_from_env()
    if slack_config.webhook_url:
        slack = SlackIntegration(slack_config)
        slack.notify_offline_message(request.user_name, request.user_email)

    return message.to_dict()


@router.get("/offline-messages", response_model=List[Dict[str, Any]])
async def get_offline_messages():
    """Get all offline messages."""
    messages = chat_service.get_offline_messages()
    return [m.to_dict() for m in messages]


@router.post("/offline-messages/{message_id}/process")
async def mark_offline_message_processed(message_id: str):
    """Mark an offline message as processed."""
    success = chat_service.mark_offline_message_processed(message_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"success": True, "message_id": message_id}


@router.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str):
    """Get transcript of a chat session."""
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.id,
        "user_name": session.user_name,
        "created_at": session.created_at.isoformat(),
        "closed_at": session.closed_at.isoformat() if session.closed_at else None,
        "messages": [m.to_dict() for m in session.messages],
    }
