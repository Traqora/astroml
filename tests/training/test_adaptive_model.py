from astroml.training.incremental.adaptive_model import AdaptiveModel
def test_adaptive_model() -> None:
    a = AdaptiveModel()
    a.adapt()
