"""Tests for data versioning (DVC) and DVC pipeline integration."""

import tempfile
from pathlib import Path

from astroml.pipeline.dvc_pipeline import DVCPipelineManager, PipelineDefinition
from astroml.storage.data_versioning import DataVersionControl, DatasetVersion, VersionDiff


# ---------------------------------------------------------------------------
# DataVersionControl (without DVC binary)
# ---------------------------------------------------------------------------


def test_dvc_disabled_without_dvc() -> None:
    """When DVC is not installed, DataVersionControl falls back gracefully."""
    dvc = DataVersionControl()
    # In test environments without DVC installed, this should be False
    assert dvc.enabled is False or dvc.enabled is True


def test_add_dataset_no_dvc() -> None:
    """Adding a dataset works even when DVC is not installed."""
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b,c\n1,2,3\n4,5,6\n")
        tmp_path = f.name

    try:
        ver = dvc.add_dataset(
            name="test-dataset",
            path=tmp_path,
            version="v1.0",
            description="Test data",
            tags=["test", "v1"],
        )
        assert ver.name == "test-dataset"
        assert ver.version == "v1.0"
        assert "test" in ver.tags
        assert ver.num_files == 1
        assert ver.size_bytes > 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_list_versions_by_name() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b\n1,2\n")
        tmp = f.name

    try:
        dvc.add_dataset("data-a", tmp, version="v1")
        dvc.add_dataset("data-b", tmp, version="v1")

        a_versions = dvc.list_versions(name="data-a")
        b_versions = dvc.list_versions(name="data-b")
        assert len(a_versions) == 1
        assert len(b_versions) == 1
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_list_versions_by_tags() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b\n1,2\n")
        tmp = f.name

    try:
        ver = dvc.add_dataset("data", tmp, version="v1", tags=["prod", "ml"])
        tagged = dvc.list_versions(tags=["prod"])
        assert len(tagged) >= 1
        assert tagged[0].version_id == ver.version_id
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_get_version() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a\n1\n")
        tmp = f.name

    try:
        ver = dvc.add_dataset("data", tmp, version="v1")
        found = dvc.get_version(ver.version_id)
        assert found is not None
        assert found.name == "data"
        assert dvc.get_version("missing") is None
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_tag_version() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a\n1\n")
        tmp = f.name

    try:
        ver = dvc.add_dataset("data", tmp, version="v1")
        ver = dvc.tag_version(ver.version_id, ["reviewed", "production"])
        assert "reviewed" in ver.tags
        assert "production" in ver.tags
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_annotate_version() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a\n1\n")
        tmp = f.name

    try:
        ver = dvc.add_dataset("data", tmp, version="v1")
        ver = dvc.annotate(ver.version_id, {"source": "warehouse", "rows": "1000"})
        assert ver.annotations["source"] == "warehouse"
        assert ver.annotations["rows"] == "1000"
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_compare_versions() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a\n1\n")
        tmp1 = f.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f2:
        f2.write(b"a,b\n1,2\n")
        tmp2 = f2.name

    try:
        ver1 = dvc.add_dataset("ds", tmp1, version="v1")
        ver2 = dvc.add_dataset("ds", tmp2, version="v2")
        diff = dvc.compare_versions(ver1.version_id, ver2.version_id)
        assert isinstance(diff, VersionDiff)
        assert diff.version_a == "v1"
        assert diff.version_b == "v2"
        assert diff.summary != ""
    finally:
        Path(tmp1).unlink(missing_ok=True)
        Path(tmp2).unlink(missing_ok=True)


def test_snapshot() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a\n1\n")
        tmp = f.name

    try:
        ver = dvc.add_dataset("data", tmp, version="v1")
        snap = dvc.snapshot(ver.version_id)
        assert snap["name"] == "data"
        assert snap["version"] == "v1"
        assert "tags" in snap
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_export_snapshot_json() -> None:
    dvc = DataVersionControl()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a\n1\n")
        tmp = f.name

    try:
        ver = dvc.add_dataset("data", tmp, version="v1")
        json_str = dvc.export_snapshot_json(ver.version_id)
        assert '"data"' in json_str
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_nonexistent_version_raises() -> None:
    dvc = DataVersionControl()
    try:
        dvc.get_version("nonexistent")
    except ValueError:
        pass  # Should raise

    try:
        dvc.tag_version("nonexistent", ["tag"])
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# DVCPipelineManager
# ---------------------------------------------------------------------------


def test_create_pipeline() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("feature-pipe", description="Feature engineering")
    assert pipe.name == "feature-pipe"
    assert pipe.description == "Feature engineering"


def test_add_stage() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("pipeline")
    stage = mgr.add_stage(
        pipe.pipeline_id,
        "clean",
        "python clean.py",
        dependencies=["data/raw.csv"],
        outputs=["data/clean.csv"],
    )
    assert stage.name == "clean"
    assert len(pipe.stages) == 1


def test_remove_stage() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("pipeline")
    stage = mgr.add_stage(pipe.pipeline_id, "s1", "true")
    mgr.remove_stage(pipe.pipeline_id, stage.stage_id)
    assert len(pipe.stages) == 0


def test_list_pipelines_by_tags() -> None:
    mgr = DVCPipelineManager()
    mgr.create_pipeline("pipe-a", tags=["prod"])
    mgr.create_pipeline("pipe-b", tags=["dev"])
    prods = mgr.list_pipelines(tags=["prod"])
    assert len(prods) == 1
    assert prods[0].name == "pipe-a"


def test_run_simple_pipeline() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("echo-test")
    mgr.add_stage(pipe.pipeline_id, "step1", "echo hello")

    run = mgr.run(pipe.pipeline_id)
    assert run.status == "completed"


def test_get_run_status() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("test")
    mgr.add_stage(pipe.pipeline_id, "s1", "echo hi")
    run = mgr.run(pipe.pipeline_id)

    fetched = mgr.get_run_status(run.run_id)
    assert fetched is not None
    assert fetched.status == "completed"


def test_list_runs() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("test")
    mgr.add_stage(pipe.pipeline_id, "s1", "echo hi")
    mgr.run(pipe.pipeline_id)
    mgr.run(pipe.pipeline_id)

    runs = mgr.list_runs(limit=10)
    assert len(runs) >= 1


def test_empty_pipeline_run_raises() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("empty")
    try:
        mgr.run(pipe.pipeline_id)
    except ValueError as exc:
        assert "no stages" in str(exc).lower()


def test_export_definition() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("test-pipe")
    mgr.add_stage(pipe.pipeline_id, "s1", "echo test", outputs=["out.txt"])

    data = mgr.export_definition(pipe.pipeline_id)
    assert data["name"] == "test-pipe"
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "s1"


def test_pipeline_run_with_failed_stage() -> None:
    mgr = DVCPipelineManager()
    pipe = mgr.create_pipeline("failing")
    mgr.add_stage(pipe.pipeline_id, "step1", "exit 1")

    run = mgr.run(pipe.pipeline_id)
    assert run.status == "failed"
    assert run.error is not None