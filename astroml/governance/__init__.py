from typing import Any, Dict, List, Optional, Union, Callable
"""Model governance and audit logging module.

Issue #637: Implements comprehensive model governance including audit logging,
approval workflows, compliance documentation, and model cards.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ModelAuditLogger",
    "AuditEvent",
    "AuditEventType",
    "ApprovalWorkflow",
    "ApprovalStatus",
    "ApprovalStep",
    "ComplianceChecker",
    "ComplianceReport",
    "ModelCard",
    "ModelCardGenerator",
    "RiskAssessment",
    "RiskLevel",
    "ModelRetirementWorkflow",
]

_LAZY: dict[str, tuple[str, str]] = {
    "ModelAuditLogger": ("astroml.governance.audit_logger", "ModelAuditLogger"),
    "AuditEvent": ("astroml.governance.audit_logger", "AuditEvent"),
    "AuditEventType": ("astroml.governance.audit_logger", "AuditEventType"),
    "ApprovalWorkflow": ("astroml.governance.approval_workflow", "ApprovalWorkflow"),
    "ApprovalStatus": ("astroml.governance.approval_workflow", "ApprovalStatus"),
    "ApprovalStep": ("astroml.governance.approval_workflow", "ApprovalStep"),
    "ComplianceChecker": ("astroml.governance.compliance", "ComplianceChecker"),
    "ComplianceReport": ("astroml.governance.compliance", "ComplianceReport"),
    "ModelCard": ("astroml.governance.model_card", "ModelCard"),
    "ModelCardGenerator": ("astroml.governance.model_card", "ModelCardGenerator"),
    "RiskAssessment": ("astroml.governance.compliance", "RiskAssessment"),
    "RiskLevel": ("astroml.governance.compliance", "RiskLevel"),
    "ModelRetirementWorkflow": (
        "astroml.governance.approval_workflow",
        "ModelRetirementWorkflow",
    ),
}


def __getattr__(name -> Any: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")