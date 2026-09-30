from astroml.tracking.experiment_tracker import ExperimentTracker
from astroml.training.hyperparameter_optimization import HPOptimizer


def test_hpo() -> None:
    HPOptimizer().optimize()
    ExperimentTracker().track()
