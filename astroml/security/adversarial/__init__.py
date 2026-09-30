"""Adversarial attack generation and defence.

Resolves part of #645.
"""

from __future__ import annotations

from astroml.security.adversarial.attacks import (
    AttackConfig,
    AttackResult,
    AttackType,
    CarliniWagnerAttack,
    FGSMAttack,
    PGDAttack,
    generate_attack,
    numerical_gradient,
)
from astroml.security.adversarial.defenses import (
    AdversarialDetector,
    DefenseType,
    DetectionResult,
    FeatureSqueezing,
    GaussianSmoothing,
    InputClipping,
    RobustnessEvaluator,
    RobustnessReport,
    adversarial_training_set,
)

__all__ = [
    "AdversarialDetector",
    "AttackConfig",
    "AttackResult",
    "AttackType",
    "CarliniWagnerAttack",
    "DefenseType",
    "DetectionResult",
    "FGSMAttack",
    "FeatureSqueezing",
    "GaussianSmoothing",
    "InputClipping",
    "PGDAttack",
    "RobustnessEvaluator",
    "RobustnessReport",
    "adversarial_training_set",
    "generate_attack",
    "numerical_gradient",
]
