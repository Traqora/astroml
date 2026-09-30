# Artifact Store Quick Reference

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Local Storage (Development)

```python
from astroml.storage import LocalArtifactStore
from astroml.tracking import MLflowTracker

store = LocalArtifactStore("./artifacts")
tracker = MLflowTracker(artifact_store=store)

# Save model
uri = tracker.log_model_artifact(model, checkpoint_path="best.pth")
```

### S3 Storage (Production)

```python
from astroml.storage import S3ArtifactStore
from astroml.tracking import MLflowTracker

store = S3ArtifactStore("my-bucket", "models")
tracker = MLflowTracker(artifact_store=store)

# Save model
uri = tracker.log_model_artifact(model, checkpoint_path="best.pth")
```

### GCS Storage (Multi-Cloud)

```python
from astroml.storage import GCSArtifactStore
from astroml.tracking import MLflowTracker

store = GCSArtifactStore("my-bucket", "models")
tracker = MLflowTracker(artifact_store=store)

# Save model
uri = tracker.log_model_artifact(model, checkpoint_path="best.pth")
```

## URI Format

```
file:///path/to/artifacts          # Local
s3://bucket-name/prefix            # S3
gs://bucket-name/prefix            # GCS
```

## Factory Function

```python
from astroml.storage import create_artifact_store

# Create from URI
store = create_artifact_store("s3://my-bucket/models")
```

## Common Operations

### Save Artifact

```python
uri = store.save("local_file.pth", "remote/path.pth")
```

### Load Artifact

```python
store.load("remote/path.pth", "local_file.pth")
```

### Check Existence

```python
if store.exists("remote/path.pth"):
    print("Artifact exists")
```

### List Artifacts

```python
artifacts = store.list_artifacts("prefix")
for artifact in artifacts:
    print(artifact)
```

### Delete Artifact

```python
store.delete("remote/path.pth")
```

### Get URI

```python
uri = store.get_uri("remote/path.pth")
print(uri)  # s3://bucket/prefix/remote/path.pth
```

## MLflow Tracker Methods

### Log Model

```python
uri = tracker.log_model_artifact(
    model=model,
    artifact_path="model",
    checkpoint_path="best.pth"
)
```

### Save Artifact

```python
uri = tracker.save_artifact(
    local_path="config.yaml",
    artifact_path="config"
)
```

### Load Artifact

```python
path = tracker.load_artifact(
    remote_path="model/best.pth",
    local_path="downloaded.pth"
)
```

## Configuration

### Local (YAML)

```yaml
artifact_storage:
  backend: local
  local:
    path: artifacts
```

### S3 (YAML)

```yaml
artifact_storage:
  backend: s3
  s3:
    bucket: my-bucket
    prefix: models
    region_name: us-east-1
```

### GCS (YAML)

```yaml
artifact_storage:
  backend: gcs
  gcs:
    bucket: my-bucket
    prefix: models
    project_id: my-project
```

## Environment Variables

### S3

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### GCS

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=my-project
```

## Hydra Integration

```python
from hydra import compose, initialize_config_dir
from astroml.storage import create_artifact_store

cfg = compose(config_name="config")
artifact_uri = cfg.training.artifact_storage.get_artifact_uri()
store = create_artifact_store(artifact_uri)
```

## Testing

```bash
# Run all tests
pytest tests/test_artifact_store.py -v

# Run specific test
pytest tests/test_artifact_store.py::TestLocalArtifactStore -v

# With coverage
pytest tests/test_artifact_store.py --cov=astroml.storage
```

## Troubleshooting

| Issue                           | Solution                                     |
| ------------------------------- | -------------------------------------------- |
| `ModuleNotFoundError: fsspec`   | `pip install -r requirements.txt`            |
| `NoCredentialsError` (S3)       | Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY |
| `DefaultCredentialsError` (GCS) | Set GOOGLE_APPLICATION_CREDENTIALS           |
| `PermissionError`               | Verify IAM permissions                       |
| `FileNotFoundError`             | Check artifact path exists                   |

## Examples

### Example 1: Save and Load

```python
from astroml.storage import LocalArtifactStore

store = LocalArtifactStore("./artifacts")

# Save
uri = store.save("model.pth", "exp1/model.pth")
print(f"Saved to: {uri}")

# Load
store.load("exp1/model.pth", "downloaded.pth")
```

### Example 2: List and Delete

```python
from astroml.storage import S3ArtifactStore

store = S3ArtifactStore("my-bucket", "models")

# List
artifacts = store.list_artifacts("exp1")
for artifact in artifacts:
    print(artifact)

# Delete old ones
for artifact in artifacts:
    if "old" in artifact:
        store.delete(artifact)
```

### Example 3: Multi-Backend

```python
import os
from astroml.storage import create_artifact_store

# Use env var to switch backends
artifact_uri = os.getenv(
    "ARTIFACT_URI",
    "file:///tmp/artifacts"
)

store = create_artifact_store(artifact_uri)
```

## API Reference

### ArtifactStore Methods

```python
# Save local file to store
uri: str = store.save(local_path, remote_path)

# Load from store to local
path: Path = store.load(remote_path, local_path)

# Check if exists
exists: bool = store.exists(remote_path)

# Delete artifact
store.delete(remote_path)

# List artifacts
artifacts: list[str] = store.list_artifacts(prefix)

# Get full URI
uri: str = store.get_uri(remote_path)
```

### MLflowTracker Methods

```python
# Log model artifact
uri: Optional[str] = tracker.log_model_artifact(
    model, artifact_path, checkpoint_path
)

# Save arbitrary artifact
uri: Optional[str] = tracker.save_artifact(
    local_path, artifact_path
)

# Load artifact
path: Path = tracker.load_artifact(
    remote_path, local_path
)
```

## Performance Tips

1. Use regional buckets for faster access
2. Compress large models before upload
3. Use multipart uploads (automatic for >100MB)
4. Cache frequently accessed artifacts locally
5. Use prefixes to organize artifacts

## Security Tips

1. Use environment variables for credentials
2. Never commit credentials to version control
3. Use IAM roles in production
4. Enable bucket versioning
5. Enable server-side encryption
6. Restrict bucket access via policies

## Documentation

- **Full Guide**: `ARTIFACT_STORAGE.md`
- **Integration**: `ARTIFACT_STORE_INTEGRATION.md`
- **Summary**: `ARTIFACT_STORE_SUMMARY.md`
- **Example**: `examples/train_with_artifact_store.py`

## Support

For issues or questions:

1. Check `ARTIFACT_STORAGE.md` troubleshooting section
2. Review example scripts
3. Run tests to verify setup
4. Check cloud provider documentation
