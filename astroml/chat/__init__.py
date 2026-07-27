"""Live chat support system for issue #306.

Provides:
- Real-time messaging via WebSocket
- Chat history and transcripts
- Agent dashboard for handling chats
- Offline message capture
- Slack integration for agents
"""
from __future__ import annotations

from .models import ChatMessage, ChatSession, AgentStatus
from .service import ChatService, chat_service
from .slack import SlackIntegration

__all__ = [
    "ChatMessage",
    "ChatSession",
    "AgentStatus",
    "ChatService",
    "chat_service",
    "SlackIntegration",
]
