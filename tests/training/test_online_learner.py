from astroml.training.incremental.online_learner import OnlineLearner
def test_online_learner() -> None:
    o = OnlineLearner()
    o.learn([])
