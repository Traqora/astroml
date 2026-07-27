"""Chat service for real-time messaging (issue #306)."""
from __future__ import annotations

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from .models import (
    ChatMessage,
    ChatSession,
    ChatStatus,
    Agent,
    AgentStatus,
    OfflineMessage,
    MessageRole,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and messages."""

    def __init__(self):
        """Initialize chat service."""
        self.sessions: Dict[str, ChatSession] = {}
        self.agents: Dict[str, Agent] = {}
        self.offline_messages: List[OfflineMessage] = []
        self.active_connections: Dict[str, set] = defaultdict(set)  # session_id -> websocket connections

    def create_session(
        self,
        user_id: str,
        user_name: str,
        user_email: Optional[str] = None,
    ) -> ChatSession:
        """Create a new chat session.

        Args:
            user_id: User identifier.
            user_name: User display name.
            user_email: Optional user email.

        Returns:
            New ChatSession.
        """
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email,
            status=ChatStatus.WAITING,
        )

        self.sessions[session_id] = session
        logger.info(f"Created chat session: {session_id} for user {user_name}")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            ChatSession if found.
        """
        return self.sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        sender_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> Optional[ChatMessage]:
        """Add a message to a chat session.

        Args:
            session_id: Session identifier.
            role: Message role (user/agent/system).
            content: Message content.
            sender_id: Optional sender identifier.
            sender_name: Optional sender name.

        Returns:
            New ChatMessage if session exists.
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
        )

        session.messages.append(message)
        logger.info(f"Added message to session {session_id}: {role.value}")
        return message

    def assign_agent(self, session_id: str, agent_id: str) -> bool:
        """Assign an agent to a chat session.

        Args:
            session_id: Session identifier.
            agent_id: Agent identifier.

        Returns:
            True if assignment successful.
        """
        session = self.sessions.get(session_id)
        agent = self.agents.get(agent_id)

        if not session or not agent:
            logger.warning(f"Cannot assign: session={session_id}, agent={agent_id}")
            return False

        if agent.status != AgentStatus.ONLINE:
            logger.warning(f"Agent {agent_id} is not online")
            return False

        if agent.current_chats >= agent.max_concurrent_chats:
            logger.warning(f"Agent {agent_id} is at capacity")
            return False

        session.assigned_agent_id = agent_id
        session.assigned_agent_name = agent.name
        session.status = ChatStatus.ACTIVE
        agent.current_chats += 1

        logger.info(f"Assigned agent {agent_id} to session {session_id}")
        return True

    def close_session(self, session_id: str) -> bool:
        """Close a chat session.

        Args:
            session_id: Session identifier.

        Returns:
            True if session closed successfully.
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = ChatStatus.CLOSED
        session.closed_at = datetime.utcnow()

        # Release agent
        if session.assigned_agent_id:
            agent = self.agents.get(session.assigned_agent_id)
            if agent:
                agent.current_chats = max(0, agent.current_chats - 1)

        logger.info(f"Closed session: {session_id}")
        return True

    def transfer_session(self, session_id: str, new_agent_id: str) -> bool:
        """Transfer a session to another agent.

        Args:
            session_id: Session identifier.
            new_agent_id: New agent identifier.

        Returns:
            True if transfer successful.
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        old_agent_id = session.assigned_agent_id

        # Release old agent
        if old_agent_id:
            old_agent = self.agents.get(old_agent_id)
            if old_agent:
                old_agent.current_chats = max(0, old_agent.current_chats - 1)

        # Assign to new agent
        return self.assign_agent(session_id, new_agent_id)

    def register_agent(
        self,
        agent_id: str,
        name: str,
        email: str,
        slack_user_id: Optional[str] = None,
        max_concurrent_chats: int = 5,
    ) -> Agent:
        """Register a support agent.

        Args:
            agent_id: Agent identifier.
            name: Agent name.
            email: Agent email.
            slack_user_id: Optional Slack user ID.
            max_concurrent_chats: Maximum concurrent chats.

        Returns:
            New Agent.
        """
        agent = Agent(
            id=agent_id,
            name=name,
            email=email,
            status=AgentStatus.OFFLINE,
            max_concurrent_chats=max_concurrent_chats,
            slack_user_id=slack_user_id,
        )

        self.agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")
        return agent

    def set_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Set agent status.

        Args:
            agent_id: Agent identifier.
            status: New status.

        Returns:
            True if status updated.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return False

        agent.status = status
        logger.info(f"Agent {agent_id} status set to {status.value}")
        return True

    def get_waiting_sessions(self) -> List[ChatSession]:
        """Get all sessions waiting for an agent.

        Returns:
            List of waiting sessions.
        """
        return [
            session for session in self.sessions.values()
            if session.status == ChatStatus.WAITING
        ]

    def get_agent_sessions(self, agent_id: str) -> List[ChatSession]:
        """Get all sessions assigned to an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            List of assigned sessions.
        """
        return [
            session for session in self.sessions.values()
            if session.assigned_agent_id == agent_id and session.status == ChatStatus.ACTIVE
        ]

    def get_online_agents(self) -> List[Agent]:
        """Get all online agents.

        Returns:
            List of online agents.
        """
        return [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.ONLINE
        ]

    def create_offline_message(
        self,
        user_name: str,
        user_email: str,
        message: str,
    ) -> OfflineMessage:
        """Create an offline message when no agents are available.

        Args:
            user_name: User name.
            user_email: User email.
            message: Message content.

        Returns:
            New OfflineMessage.
        """
        offline_msg = OfflineMessage(
            id=str(uuid.uuid4()),
            user_name=user_name,
            user_email=user_email,
            message=message,
        )

        self.offline_messages.append(offline_msg)
        logger.info(f"Created offline message from {user_name}")
        return offline_msg

    def get_offline_messages(self) -> List[OfflineMessage]:
        """Get all offline messages.

        Returns:
            List of offline messages.
        """
        return self.offline_messages

    def mark_offline_message_processed(self, message_id: str) -> bool:
        """Mark an offline message as processed.

        Args:
            message_id: Message identifier.

        Returns:
            True if marked.
        """
        for msg in self.offline_messages:
            if msg.id == message_id:
                msg.is_processed = True
                return True
        return False

    def add_connection(self, session_id: str, connection_id: str) -> None:
        """Add a WebSocket connection to a session.

        Args:
            session_id: Session identifier.
            connection_id: Connection identifier.
        """
        self.active_connections[session_id].add(connection_id)

    def remove_connection(self, session_id: str, connection_id: str) -> None:
        """Remove a WebSocket connection from a session.

        Args:
            session_id: Session identifier.
            connection_id: Connection identifier.
        """
        self.active_connections[session_id].discard(connection_id)

    def get_session_connections(self, session_id: str) -> set:
        """Get all connections for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Set of connection IDs.
        """
        return self.active_connections.get(session_id, set())


# Global chat service instance
chat_service = ChatService()
