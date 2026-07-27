"""Slack integration for chat support (issue #306)."""
from __future__ import annotations

import logging
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SlackConfig:
    """Configuration for Slack integration."""

    webhook_url: Optional[str] = None
    bot_token: Optional[str] = None
    channel: str = "#support"


class SlackIntegration:
    """Integration with Slack for agent notifications."""

    def __init__(self, config: SlackConfig):
        """Initialize Slack integration.

        Args:
            config: Slack configuration.
        """
        self.config = config

    def send_webhook(self, message: str) -> bool:
        """Send a message via Slack webhook.

        Args:
            message: Message to send.

        Returns:
            True if sent successfully.
        """
        if not self.config.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        try:
            payload = {"text": message}
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("Slack webhook sent successfully")
                return True
            else:
                logger.error(f"Slack webhook failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Slack webhook error: {e}")
            return False

    def notify_new_chat(self, user_name: str, session_id: str) -> bool:
        """Notify agents about a new chat session.

        Args:
            user_name: User name.
            session_id: Session identifier.

        Returns:
            True if notification sent.
        """
        message = f"🆕 New chat from {user_name}\nSession ID: {session_id}"
        return self.send_webhook(message)

    def notify_agent_assigned(self, agent_name: str, session_id: str) -> bool:
        """Notify about agent assignment.

        Args:
            agent_name: Agent name.
            session_id: Session identifier.

        Returns:
            True if notification sent.
        """
        message = f"✅ Agent {agent_name} assigned to session {session_id}"
        return self.send_webhook(message)

    def notify_offline_message(self, user_name: str, user_email: str) -> bool:
        """Notify about offline message.

        Args:
            user_name: User name.
            user_email: User email.

        Returns:
            True if notification sent.
        """
        message = f"📩 Offline message from {user_name} ({user_email})"
        return self.send_webhook(message)

    def send_direct_message(self, user_id: str, message: str) -> bool:
        """Send a direct message to a Slack user.

        Args:
            user_id: Slack user ID.
            message: Message to send.

        Returns:
            True if sent successfully.
        """
        if not self.config.bot_token:
            logger.warning("Slack bot token not configured")
            return False

        try:
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.config.bot_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "channel": user_id,
                "text": message,
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    logger.info(f"Direct message sent to {user_id}")
                    return True
                else:
                    logger.error(f"Slack API error: {data.get('error')}")
                    return False
            else:
                logger.error(f"Slack API request failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Slack direct message error: {e}")
            return False

    def create_slack_config_from_env() -> SlackConfig:
        """Create Slack config from environment variables."""
        import os

        return SlackConfig(
            webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
            bot_token=os.environ.get("SLACK_BOT_TOKEN"),
            channel=os.environ.get("SLACK_CHANNEL", "#support"),
        )
