"""Automated model calibration and uncertainty estimation toolkit.

Provides probability calibration (Platt scaling, isotonic regression),
conformal prediction, and Bayesian-style uncertainty estimation helpers.
"""

from astroml.training.calibration.bayesian import BayesianUncertainty
from astroml.training.calibration.conformal import (
    ConformalPredictor,
    ConformalResult,
)
from astroml.training.calibration.isotonic import IsotonicCalibrator
from astroml.training.calibration.platt import PlattCalibrator

__all__ = [
    "BayesianUncertainty",
    "ConformalPredictor",
    "ConformalResult",
    "IsotonicCalibrator",
    "PlattCalibrator",
]
