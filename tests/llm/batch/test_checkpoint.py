"""Tests for checkpoint manager."""

from astroml.llm.batch.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_initial_state(self):
        c = CheckpointManager("job_1")
        progress = c.get_progress()
        assert progress["processed"] == 0
        assert progress["failed"] == 0
        assert progress["last_position"] is None

    def test_save_and_load_position(self):
        c = CheckpointManager("job_1")
        c.save("100")
        assert c.load() == 100

    def test_record_success(self):
        c = CheckpointManager("job_1")
        c.record_success(5)
        assert c.get_progress()["processed"] == 5

    def test_record_failure(self):
        c = CheckpointManager("job_1")
        c.record_failure(3)
        assert c.get_progress()["failed"] == 3

    def test_to_json_and_from_json(self):
        c = CheckpointManager("job_1")
        c.save("50")
        c.record_success(10)
        c.record_failure(2)

        raw = c.to_json()
        restored = CheckpointManager.from_json("job_1", raw)
        loaded = restored.load()
        assert loaded == 50 or loaded == "50"
        assert restored.get_progress()["processed"] == 10
        assert restored.get_progress()["failed"] == 2

    def test_from_json_empty(self):
        restored = CheckpointManager.from_json("job_1", "")
        assert restored.load() is None
        assert restored.get_progress()["processed"] == 0
