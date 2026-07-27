"""Alert configuration and threshold checks for LLM costs."""
from __future__ import annotations

import logging
from sqlalchemy.ext.asyncio import AsyncSession
from astroml.db.models.cost import LLMBudget

logger = logging.getLogger(__name__)


async def check_and_trigger_alerts(db: AsyncSession, budget: LLMBudget) -> None:
    """
    Check if spend has crossed thresholds (50%, 80%, 100%) and trigger warnings/notifications.
    Prevents duplicate alerts by tracking the last_alert_threshold.
    """
    if budget.limit_amount <= 0:
        return
        
    ratio = budget.current_spend / budget.limit_amount
    
    # Thresholds: 1.0 (100%), 0.8 (80%), 0.5 (50%)
    thresholds = [1.0, 0.8, 0.5]
    
    for t in thresholds:
        if ratio >= t:
            # Check if this threshold has already been alerted
            if budget.last_alert_threshold < t:
                # Update threshold first to prevent race condition/duplicate alerts
                budget.last_alert_threshold = t
                
                percent = int(t * 100)
                message = (
                    f"LLM Cost Alert! Budget threshold {percent}% crossed for {budget.scope} "
                    f"'{budget.entity_id}'. Spend: ${budget.current_spend:.2f} / limit: ${budget.limit_amount:.2f}."
                )
                
                if percent >= 100:
                    logger.error(message)
                else:
                    logger.warning(message)
                    
                # Optionally insert a system notification if the entity_id is numeric (represents user ID)
                try:
                    # Try local import to avoid circular dependency
                    from api.models.orm import Notification
                    
                    if budget.scope == "user" and budget.entity_id.isdigit():
                        user_id_int = int(budget.entity_id)
                        notif = Notification(
                            user_id=user_id_int,
                            event_type="cost_alert",
                            title=f"LLM Budget {percent}% Limit Crossed",
                            content=message,
                            is_read=False
                        )
                        db.add(notif)
                except Exception as e:
                    logger.debug("Could not create ORM notification: %s", e)
                    
                break  # only alert for the highest crossed threshold
