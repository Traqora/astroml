"""Tests for artifact storage backends."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astroml.storage import (
    ArtifactStorageConfig,
    GCSArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    create_artifact_store,
)


class TestLocalArtifactStore:
    """Tests for local filesystem artifact store."""

    def test_init_creates_directory(self):
        """Test that initialization creates base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)
            assert Path(tmpdir).exists()

    def test_save_and_load(self):
        """Test saving and loading artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            # Save artifact
            uri = store.save(test_file, "artifacts/test.txt")
            assert uri.startswith("file://")
            assert store.exists("artifacts/test.txt")

            # Load artifact
            load_path = Path(tmpdir) / "loaded.txt"
            loaded = store.load("artifacts/test.txt", load_path)
            assert loaded.read_text() == "test content"

    def test_exists(self):
        """Test checking artifact existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)

            # Create and save artifact
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            store.save(test_file, "test.txt")

            assert store.exists("test.txt")
            assert not store.exists("nonexistent.txt")

    def test_delete(self):
        """Test deleting artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)

            # Create and save artifact
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            store.save(test_file, "test.txt")

            assert store.exists("test.txt")
            store.delete("test.txt")
            assert not store.exists("test.txt")

    def test_list_artifacts(self):
        """Test listing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)

            # Create and save multiple artifacts
            for i in range(3):
                test_file = Path(tmpdir) / f"test{i}.txt"
                test_file.write_text(f"test {i}")
                store.save(test_file, f"test{i}.txt")

            artifacts = store.list_artifacts()
            assert len(artifacts) == 3

    def test_get_uri(self):
        """Test getting artifact URI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)
            uri = store.get_uri("test.txt")
            assert uri.startswith("file://")
            assert "test.txt" in uri

    def test_save_nonexistent_file_raises(self):
        """Test that saving nonexistent file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)
            with pytest.raises(FileNotFoundError):
                store.save("nonexistent.txt", "test.txt")

    def test_load_nonexistent_artifact_raises(self):
        """Test that loading nonexistent artifact raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(tmpdir)
            with pytest.raises(FileNotFoundError):
                store.load("nonexistent.txt", "local.txt")


class TestS3ArtifactStore:
    """Tests for S3 artifact store."""

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_init(self, mock_fs):
        """Test S3 store initialization."""
        store = S3ArtifactStore("my-bucket", "prefix")
        assert store.bucket == "my-bucket"
        assert store.prefix == "prefix"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_get_s3_path(self, mock_fs):
        """Test S3 path construction."""
        store = S3ArtifactStore("my-bucket", "prefix")
        path = store._get_s3_path("test.txt")
        assert path == "my-bucket/prefix/test.txt"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_get_s3_path_no_prefix(self, mock_fs):
        """Test S3 path construction without prefix."""
        store = S3ArtifactStore("my-bucket")
        path = store._get_s3_path("test.txt")
        assert path == "my-bucket/test.txt"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_get_uri(self, mock_fs):
        """Test getting S3 URI."""
        store = S3ArtifactStore("my-bucket", "prefix")
        uri = store.get_uri("test.txt")
        assert uri == "s3://my-bucket/prefix/test.txt"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_save(self, mock_fs):
        """Test saving to S3."""
        mock_fs_instance = MagicMock()
        mock_fs.return_value = mock_fs_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            store = S3ArtifactStore("my-bucket", "prefix")
            uri = store.save(test_file, "test.txt")

            assert uri == "s3://my-bucket/prefix/test.txt"
            mock_fs_instance.put.assert_called_once()

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_load(self, mock_fs):
        """Test loading from S3."""
        mock_fs_instance = MagicMock()
        mock_fs.return_value = mock_fs_instance
        mock_fs_instance.exists.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            store = S3ArtifactStore("my-bucket", "prefix")
            local_path = store.load("test.txt", Path(tmpdir) / "local.txt")

            assert local_path.parent.exists()
            mock_fs_instance.get.assert_called_once()

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_exists(self, mock_fs):
        """Test checking S3 artifact existence."""
        mock_fs_instance = MagicMock()
        mock_fs.return_value = mock_fs_instance
        mock_fs_instance.exists.return_value = True

        store = S3ArtifactStore("my-bucket", "prefix")
        assert store.exists("test.txt")
        mock_fs_instance.exists.assert_called_once()

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_delete(self, mock_fs):
        """Test deleting from S3."""
        mock_fs_instance = MagicMock()
        mock_fs.return_value = mock_fs_instance
        mock_fs_instance.exists.return_value = True

        store = S3ArtifactStore("my-bucket", "prefix")
        store.delete("test.txt")
        mock_fs_instance.rm.assert_called_once()


class TestGCSArtifactStore:
    """Tests for GCS artifact store."""

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_init(self, mock_fs):
        """Test GCS store initialization."""
        store = GCSArtifactStore("my-bucket", "prefix")
        assert store.bucket == "my-bucket"
        assert store.prefix == "prefix"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_get_gcs_path(self, mock_fs):
        """Test GCS path construction."""
        store = GCSArtifactStore("my-bucket", "prefix")
        path = store._get_gcs_path("test.txt")
        assert path == "my-bucket/prefix/test.txt"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_get_uri(self, mock_fs):
        """Test getting GCS URI."""
        store = GCSArtifactStore("my-bucket", "prefix")
        uri = store.get_uri("test.txt")
        assert uri == "gs://my-bucket/prefix/test.txt"


class TestCreateArtifactStore:
    """Tests for artifact store factory function."""

    def test_create_local_store(self):
        """Test creating local artifact store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_artifact_store(f"file://{tmpdir}")
            assert isinstance(store, LocalArtifactStore)

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_create_s3_store(self, mock_fs):
        """Test creating S3 artifact store."""
        store = create_artifact_store("s3://my-bucket/prefix")
        assert isinstance(store, S3ArtifactStore)
        assert store.bucket == "my-bucket"
        assert store.prefix == "prefix"

    @patch("astroml.storage.artifact_store.fsspec.filesystem")
    def test_create_gcs_store(self, mock_fs):
        """Test creating GCS artifact store."""
        store = create_artifact_store("gs://my-bucket/prefix")
        assert isinstance(store, GCSArtifactStore)
        assert store.bucket == "my-bucket"
        assert store.prefix == "prefix"

    def test_create_invalid_uri_raises(self):
        """Test that invalid URI raises error."""
        with pytest.raises(ValueError):
            create_artifact_store("invalid://bucket/path")


class TestArtifactStorageConfig:
    """Tests for artifact storage configuration."""

    def test_local_config(self):
        """Test local storage configuration."""
        config = ArtifactStorageConfig(backend="local")
        uri = config.get_artifact_uri()
        assert uri.startswith("file://")

    def test_s3_config(self):
        """Test S3 storage configuration."""
        config = ArtifactStorageConfig(
            backend="s3",
            s3={"bucket": "my-bucket", "prefix": "models"},
        )
        uri = config.get_artifact_uri()
        assert uri == "s3://my-bucket/models"

    def test_gcs_config(self):
        """Test GCS storage configuration."""
        config = ArtifactStorageConfig(
            backend="gcs",
            gcs={"bucket": "my-bucket", "prefix": "models"},
        )
        uri = config.get_artifact_uri()
        assert uri == "gs://my-bucket/models"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ArtifactStorageConfig(backend="local")
        config_dict = config.to_dict()
        assert config_dict["backend"] == "local"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {"backend": "local", "local": {"path": "artifacts"}}
        config = ArtifactStorageConfig.from_dict(config_dict)
        assert config.backend == "local"
