"""Chat models for issue #306."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass, field


class MessageRole(Enum):
    """Role of the message sender."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class ChatStatus(Enum):
    """Status of a chat session."""

    ACTIVE = "active"
    WAITING = "waiting"
    CLOSED = "closed"
    TRANSFERRED = "transferred"


class AgentStatus(Enum):
    """Status of a support agent."""

    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
    AWAY = "away"


@dataclass
class ChatMessage:
    """A single chat message."""

    id: str
    session_id: str
    role: MessageRole
    content: str
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_read: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value,
            "content": self.content,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "timestamp": self.timestamp.isoformat(),
            "is_read": self.is_read,
        }


@dataclass
class ChatSession:
    """A chat session between a user and agent."""

    id: str
    user_id: str
    user_name: str
    user_email: Optional[str] = None
    status: ChatStatus = ChatStatus.WAITING
    assigned_agent_id: Optional[str] = None
    assigned_agent_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "assigned_agent_name": self.assigned_agent_name,
            "created_at": self.created_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }


@dataclass
class Agent:
    """A support agent."""

    id: str
    name: str
    email: str
    status: AgentStatus = AgentStatus.OFFLINE
    max_concurrent_chats: int = 5
    current_chats: int = 0
    slack_user_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "status": self.status.value,
            "max_concurrent_chats": self.max_concurrent_chats,
            "current_chats": self.current_chats,
            "slack_user_id": self.slack_user_id,
        }


@dataclass
class OfflineMessage:
    """Message captured when agents are offline."""

    id: str
    user_name: str
    user_email: str
    message: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_processed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "is_processed": self.is_processed,
        }
