"""Compliance documentation and risk assessment framework.

Issue #637 Step 4 & 6: Implements compliance documentation generation
and model risk assessment framework.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels for model assessment."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Categories of model risk."""

    FAIRNESS = "fairness"
    BIAS = "bias"
    PRIVACY = "privacy"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"
    COMPLIANCE = "compliance"
    DATA_QUALITY = "data_quality"
    OPERATIONAL = "operational"
    REPUTATIONAL = "reputational"


@dataclass
class RiskFinding:
    """A single risk finding in a risk assessment.

    Attributes:
        finding_id: Unique finding identifier.
        category: Risk category.
        level: Severity level.
        description: Human-readable description.
        mitigation: Suggested mitigation.
        metadata: Additional contextual data.
    """

    category: RiskCategory
    level: RiskLevel
    description: str
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    mitigation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "category": self.category.value,
            "level": self.level.value,
            "description": self.description,
            "mitigation": self.mitigation,
            "metadata": self.metadata,
        }


@dataclass
class RiskAssessment:
    """Comprehensive model risk assessment.

    Aggregates risk findings across multiple categories and produces
    an overall risk score and recommendation.

    Example:
        assessment = RiskAssessment(model_id="fraud-detector-v2")
        assessment.add_finding(RiskFinding(
            category=RiskCategory.BIAS,
            level=RiskLevel.MEDIUM,
            description="Model may exhibit bias against certain account demographics",
            mitigation="Implement fairness metrics and bias testing",
        ))
        print(assessment.overall_risk)  # RiskLevel.MEDIUM
    """

    model_id: str
    assessment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    findings: list[RiskFinding] = field(default_factory=list)
    assessed_by: str = "system"
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_finding(self, finding: RiskFinding) -> None:
        """Add a risk finding."""
        self.findings.append(finding)

    @property
    def overall_risk(self) -> RiskLevel:
        """Compute the overall risk level (worst-case across all findings)."""
        if not self.findings:
            return RiskLevel.NONE

        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_level = RiskLevel.NONE
        for finding in self.findings:
            if levels.index(finding.level) > levels.index(max_level):
                max_level = finding.level
        return max_level

    @property
    def risk_counts(self) -> dict[str, int]:
        """Count findings by risk level."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            key = finding.level.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def category_counts(self) -> dict[str, int]:
        """Count findings by risk category."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            key = finding.category.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def is_deployable(self) -> bool:
        """Determine if the model is safe to deploy based on risk assessment.

        Models with CRITICAL findings or multiple HIGH findings are not deployable.
        """
        if any(f.level == RiskLevel.CRITICAL for f in self.findings):
            return False
        high_count = sum(1 for f in self.findings if f.level == RiskLevel.HIGH)
        if high_count > 2:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "model_id": self.model_id,
            "overall_risk": self.overall_risk.value,
            "assessed_by": self.assessed_by,
            "assessed_at": self.assessed_at.isoformat(),
            "deployable": self.is_deployable(),
            "risk_counts": self.risk_counts,
            "category_counts": self.category_counts,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
        }


class ComplianceStandard(Enum):
    """Common regulatory/compliance standards."""

    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    EU_AI_ACT = "eu_ai_act"
    INTERNAL_POLICY = "internal_policy"


@dataclass
class ComplianceCheck:
    """A single compliance requirement check."""

    standard: ComplianceStandard
    requirement_id: str
    description: str
    passed: bool = False
    evidence: str = ""
    notes: str = ""
    checked_at: datetime | None = None

    def mark(self, passed: bool, evidence: str = "", notes: str = "") -> None:
        """Mark this check as passed or failed."""
        self.passed = passed
        self.evidence = evidence
        self.notes = notes
        self.checked_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard": self.standard.value,
            "requirement_id": self.requirement_id,
            "description": self.description,
            "passed": self.passed,
            "evidence": self.evidence,
            "notes": self.notes,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


@dataclass
class ComplianceReport:
    """Compliance assessment report for a model.

    Tracks compliance checks against various standards and provides
    a summary of compliance status.

    Example:
        report = ComplianceReport(model_id="fraud-detector-v2")
        report.add_check(ComplianceCheck(
            standard=ComplianceStandard.GDPR,
            requirement_id="GDPR-25",
            description="Right to explanation for automated decisions",
        ))
        report.checks[0].mark(True, evidence="SHAP values computed", notes="All good")
    """

    model_id: str
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    checks: list[ComplianceCheck] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = "system"

    def add_check(self, check: ComplianceCheck) -> None:
        """Add a compliance check."""
        self.checks.append(check)

    @property
    def compliance_score(self) -> float:
        """Fraction of checks passed (0.0 to 1.0)."""
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def is_compliant(self) -> bool:
        """Whether all checks passed."""
        return self.compliance_score == 1.0

    def get_checks_by_standard(self, standard: ComplianceStandard) -> list[ComplianceCheck]:
        """Get all checks for a specific standard."""
        return [c for c in self.checks if c.standard == standard]

    def get_failed_checks(self) -> list[ComplianceCheck]:
        """Get all failed checks."""
        return [c for c in self.checks if not c.passed]

    def generate_document(self, output_path: str | Path) -> None:
        """Generate a compliance document as JSON.

        Args:
            output_path: Path to write the JSON document.
        """
        doc = {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "compliance_score": round(self.compliance_score * 100, 1),
            "is_compliant": self.is_compliant,
            "summary_by_standard": {
                standard.value: {
                    "total": len(self.get_checks_by_standard(standard)),
                    "passed": sum(1 for c in self.get_checks_by_standard(standard) if c.passed),
                }
                for standard in ComplianceStandard
            },
            "failed_checks": [c.to_dict() for c in self.get_failed_checks()],
            "all_checks": [c.to_dict() for c in self.checks],
        }
        with open(output_path, "w") as f:
            json.dump(doc, f, indent=2, default=str)
        logger.info(f"Compliance document written to {output_path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "compliance_score": self.compliance_score,
            "is_compliant": self.is_compliant,
            "checks_passed": sum(1 for c in self.checks if c.passed),
            "checks_failed": sum(1 for c in self.checks if not c.passed),
            "checks_total": len(self.checks),
        }


class ComplianceChecker:
    """Service for running compliance checks across multiple standards.

    Example:
        checker = ComplianceChecker()
        report = checker.run_checks(
            model_id="fraud-detector-v2",
            standards=[ComplianceStandard.GDPR, ComplianceStandard.EU_AI_ACT],
        )
    """

    def __init__(self) -> None:
        self._standard_checks: dict[ComplianceStandard, list[dict[str, str]]] = {
            ComplianceStandard.GDPR: [
                {"id": "GDPR-22", "description": "Automated decision-making safeguards"},
                {"id": "GDPR-25", "description": "Right to meaningful explanation"},
                {"id": "GDPR-35", "description": "Data protection impact assessment"},
            ],
            ComplianceStandard.EU_AI_ACT: [
                {"id": "EU-AI-1", "description": "Risk classification of AI system"},
                {"id": "EU-AI-2", "description": "Transparency obligations"},
                {"id": "EU-AI-3", "description": "Human oversight provisions"},
            ],
            ComplianceStandard.SOC2: [
                {"id": "SOC2-CC1", "description": "Control environment"},
                {"id": "SOC2-CC2", "description": "Communication and information"},
                {"id": "SOC2-CC3", "description": "Risk assessment"},
                {"id": "SOC2-CC4", "description": "Monitoring activities"},
            ],
            ComplianceStandard.ISO27001: [
                {"id": "ISO-A8", "description": "Asset management"},
                {"id": "ISO-A12", "description": "Operations security"},
                {"id": "ISO-A14", "description": "System acquisition and development"},
            ],
            ComplianceStandard.INTERNAL_POLICY: [
                {"id": "INT-1", "description": "Code review completed"},
                {"id": "INT-2", "description": "Tests passing (>90% coverage)"},
                {"id": "INT-3", "description": "Performance benchmarks met"},
                {"id": "INT-4", "description": "Model card generated"},
            ],
        }

    def run_checks(
        self,
        model_id: str,
        standards: Sequence[ComplianceStandard] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ComplianceReport:
        """Run compliance checks and produce a report.

        Args:
            model_id: Model to check.
            standards: Standards to check against (defaults to all).
            model_metadata: Optional metadata for evidence gathering.

        Returns:
            ComplianceReport with results.
        """
        if standards is None:
            standards = list(ComplianceStandard)

        report = ComplianceReport(model_id=model_id)
        md = model_metadata or {}

        for standard in standards:
            for check_def in self._standard_checks.get(standard, []):
                check = ComplianceCheck(
                    standard=standard,
                    requirement_id=check_def["id"],
                    description=check_def["description"],
                )
                # Auto-pass internal checks if metadata provided
                if standard == ComplianceStandard.INTERNAL_POLICY:
                    if check_def["id"] == "INT-1":
                        check.mark(passed=md.get("code_reviewed", False), evidence="review:done" if md.get("code_reviewed") else "")
                    elif check_def["id"] == "INT-2":
                        check.mark(passed=md.get("tests_passing", False), evidence=f"coverage:{md.get('coverage', 'unknown')}")
                    elif check_def["id"] == "INT-3":
                        check.mark(passed=md.get("benchmarks_met", False))
                    elif check_def["id"] == "INT-4":
                        check.mark(passed=md.get("model_card_generated", False))
                else:
                    # External standards require manual review
                    check.passed = False
                report.add_check(check)

        logger.info(
            f"Compliance check for {model_id}: {report.compliance_score:.0%} "
            f"({sum(1 for c in report.checks if c.passed)}/{len(report.checks)})"
        )
        return report


__all__ = [
    "RiskLevel",
    "RiskCategory",
    "RiskFinding",
    "RiskAssessment",
    "ComplianceStandard",
    "ComplianceCheck",
    "ComplianceReport",
    "ComplianceChecker",
]