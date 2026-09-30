from astroml.training.pipeline import RetrainingPipeline
from astroml.validation.drift_detector import DriftDetector


def test_pipeline() -> None:
    RetrainingPipeline().trigger()
    assert DriftDetector().detect() is False
