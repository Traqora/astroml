"""API v1 routers package."""
from api.routers.v1.accounts import router as accounts_router
from api.routers.v1.audit import router as audit_router
from api.routers.v1.auth import router as auth_router
from api.routers.v1.backup import router as backup_router
from api.routers.v1.chat import router as chat_router
from api.routers.v1.compliance import router as compliance_router
from api.routers.v1.contact import router as contact_router
from api.routers.v1.errors import router as errors_router
from api.routers.v1.faq import router as faq_router
from api.routers.v1.feedback import router as feedback_router
from api.routers.v1.fraud import router as fraud_router
from api.routers.v1.loyalty import router as loyalty_router
from api.routers.v1.mentorship import router as mentorship_router
from api.routers.v1.models import router as models_router
from api.routers.v1.monitoring import router as monitoring_router
from api.routers.v1.notifications import router as notifications_router
from api.routers.v1.contributors import router as contributors_router
from api.routers.v1.discussions import router as discussions_router
from api.routers.v1.rate_limit import router as rate_limit_router
from api.routers.v1.transactions import router as transactions_router
from api.routers.v1.onboarding import router as onboarding_router
from api.routers.v1.validation import router as validation_router
from api.routers.v1.voice import router as voice_router
from api.routers.v1.ws import router as ws_router
from api.routers.v1.streaming import router as streaming_router
from api.routers.v1.llm import router as llm_router
from api.routers.v1.llm_health import router as llm_health_router
from api.routers.v1.reports import router as reports_router
from api.routers.v1.alerts import router as alerts_router
from api.routers.v1.query import router as query_router
from api.routers.v1.explanations import router as explanations_router
from api.routers.v1.agents import router as agents_router
from api.routers.v1.cost import router as cost_router
from api.routers.v1.llm_usage import router as llm_usage_router
from api.routers.v1.llm_cache_metrics import router as llm_cache_metrics_router
from api.routers.v1.llm_metrics import router as llm_metrics_router
from api.routers.v1.search import router as search_router
from api.routers.v1.stream import router as stream_router

__all__ = [
    "accounts_router", "audit_router", "auth_router", "backup_router",
    "chat_router", "compliance_router", "contact_router", "errors_router",
    "faq_router", "feedback_router", "fraud_router", "loyalty_router",
    "mentorship_router", "models_router", "monitoring_router",
    "notifications_router", "contributors_router", "discussions_router",
    "rate_limit_router", "transactions_router", "onboarding_router",
    "validation_router", "voice_router", "ws_router", "streaming_router",
    "llm_router", "llm_health_router", "reports_router", "alerts_router",
    "query_router", "explanations_router", "agents_router", "cost_router",
    "llm_usage_router", "llm_cache_metrics_router", "llm_metrics_router",
    "search_router", "stream_router",
]
