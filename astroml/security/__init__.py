"""ML model security: adversarial robustness, extraction and poisoning defence.

Resolves #645.

Submodules
----------
``adversarial``
    FGSM / PGD / Carlini-Wagner attacks and input-space defences.
``model_extraction``
    Behavioural detection of model-stealing query patterns.
``poisoning_detection``
    Training-data poisoning screening (label flips, outliers, backdoors).
``scoring``
    Composite security score, test pipeline and report generation.

Symbols are resolved lazily so importing one area does not pull in the others.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AdversarialDetector",
    "AttackConfig",
    "AttackType",
    "ExtractionRisk",
    "ModelExtractionDetector",
    "PoisoningDetector",
    "PoisoningReport",
    "RobustnessEvaluator",
    "SecurityScore",
    "SecurityTestPipeline",
    "SecurityTestResult",
    "generate_attack",
]

_LAZY: dict[str, tuple[str, str]] = {
    "AdversarialDetector": ("astroml.security.adversarial.defenses", "AdversarialDetector"),
    "AttackConfig": ("astroml.security.adversarial.attacks", "AttackConfig"),
    "AttackType": ("astroml.security.adversarial.attacks", "AttackType"),
    "ExtractionRisk": ("astroml.security.model_extraction", "ExtractionRisk"),
    "ModelExtractionDetector": (
        "astroml.security.model_extraction",
        "ModelExtractionDetector",
    ),
    "PoisoningDetector": ("astroml.security.poisoning_detection", "PoisoningDetector"),
    "PoisoningReport": ("astroml.security.poisoning_detection", "PoisoningReport"),
    "RobustnessEvaluator": ("astroml.security.adversarial.defenses", "RobustnessEvaluator"),
    "SecurityScore": ("astroml.security.scoring", "SecurityScore"),
    "SecurityTestPipeline": ("astroml.security.scoring", "SecurityTestPipeline"),
    "SecurityTestResult": ("astroml.security.scoring", "SecurityTestResult"),
    "generate_attack": ("astroml.security.adversarial.attacks", "generate_attack"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from its owning submodule on first access."""
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        value = getattr(import_module(module_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
