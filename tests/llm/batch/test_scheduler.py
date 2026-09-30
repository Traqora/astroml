"""Tests for backfill scheduler."""

from astroml.llm.batch.scheduler import BackfillScheduler


class TestBackfillScheduler:
    def setup_method(self):
        self.scheduler = BackfillScheduler()

    def test_create_job(self):
        job = self.scheduler.create_job("embedding", 500, {"source": "test"})
        assert job["job_type"] == "embedding"
        assert job["total_items"] == 500
        assert job["status"] == "pending"

    def test_get_job(self):
        created = self.scheduler.create_job("label", 100)
        fetched = self.scheduler.get_job(created["id"])
        assert fetched is not None
        assert fetched["id"] == created["id"]

    def test_get_job_not_found(self):
        assert self.scheduler.get_job("nonexistent") is None

    def test_list_jobs(self):
        self.scheduler.create_job("embedding", 10)
        self.scheduler.create_job("explanation", 20)
        assert len(self.scheduler.list_jobs()) == 2

    def test_pause_and_resume(self):
        job = self.scheduler.create_job("report", 50)
        paused = self.scheduler.pause_job(job["id"])
        assert paused["status"] == "paused"
        resumed = self.scheduler.resume_job(job["id"])
        assert resumed["status"] == "running"

    def test_update_job(self):
        job = self.scheduler.create_job("embedding", 100)
        updated = self.scheduler.update_job(job["id"], {"processed_items": 50})
        assert updated["processed_items"] == 50
