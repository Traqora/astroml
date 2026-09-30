from astroml.training.incremental.stream_trainer import StreamTrainer
def test_stream_trainer() -> None:
    s = StreamTrainer()
    s.train()
