"""Tests for batching strategies."""

from astroml.llm.batch.strategies import FixedSizeStrategy, AdaptiveStrategy


class TestFixedSizeStrategy:
    def test_fixed_size_never_changes(self):
        s = FixedSizeStrategy(50)
        assert s.get_batch_size() == 50
        s.on_success()
        assert s.get_batch_size() == 50
        s.on_failure(RuntimeError("x"))
        assert s.get_batch_size() == 50

    def test_reset_returns_to_initial(self):
        s = FixedSizeStrategy(25)
        s.on_success()
        assert s.get_batch_size() == 25


class TestAdaptiveStrategy:
    def test_starts_at_max(self):
        s = AdaptiveStrategy(10, 100)
        assert s.get_batch_size() == 100

    def test_shrinks_on_failure(self):
        s = AdaptiveStrategy(10, 100, shrink_factor=0.5, grow_factor=1.1)
        s.on_failure(RuntimeError("fail"))
        assert s.get_batch_size() == 50

    def test_grows_on_success(self):
        s = AdaptiveStrategy(10, 100, shrink_factor=0.5, grow_factor=1.1)
        s.on_success()
        assert s.get_batch_size() == 100  # already at max

    def test_shrink_then_grow(self):
        s = AdaptiveStrategy(2, 50, shrink_factor=0.5, grow_factor=2.0)
        s.on_failure(RuntimeError("fail"))
        assert s.get_batch_size() == 25
        s.on_success()
        assert s.get_batch_size() == 50

    def test_does_not_below_min(self):
        s = AdaptiveStrategy(5, 100, shrink_factor=0.1, grow_factor=1.1)
        for _ in range(10):
            s.on_failure(RuntimeError("fail"))
        assert s.get_batch_size() >= 5

    def test_reset_restores_max(self):
        s = AdaptiveStrategy(5, 50, shrink_factor=0.5)
        s.on_failure(RuntimeError("fail"))
        assert s.get_batch_size() == 25
        s.reset()
        assert s.get_batch_size() == 50
