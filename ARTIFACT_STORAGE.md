# Artifact Storage Configuration

AstroML now supports configurable artifact storage backends for model artifacts and other training outputs. This allows you to save models to local filesystem, AWS S3, or Google Cloud Storage (GCS) seamlessly.

## Overview

The artifact storage system provides:

- **Multiple backends**: Local filesystem, AWS S3, Google Cloud Storage
- **Unified interface**: Same API regardless of backend
- **fsspec integration**: Leverages fsspec for robust cloud storage handling
- **MLflow integration**: Seamlessly logs artifacts to both MLflow and your configured store
- **Configuration-driven**: Define storage backend via YAML config

## Quick Start

### Local Storage (Default)

```yaml
# configs/artifact_storage/local.yaml
artifact_storage:
  backend: local
  local:
    path: artifacts
```

### AWS S3

```yaml
# configs/artifact_storage/s3.yaml
artifact_storage:
  backend: s3
  s3:
    bucket: my-astroml-bucket
    prefix: models
    region_name: us-east-1
```

### Google Cloud Storage

```yaml
# configs/artifact_storage/gcs.yaml
artifact_storage:
  backend: gcs
  gcs:
    bucket: my-astroml-bucket
    prefix: models
    project_id: my-gcp-project
```

## Configuration

### Local Storage

```yaml
artifact_storage:
  backend: local
  local:
    path: /path/to/artifacts # Base directory for artifacts
```

**Environment Variables**: None required

### AWS S3

```yaml
artifact_storage:
  backend: s3
  s3:
    bucket: my-bucket # S3 bucket name (required)
    prefix: models # Optional prefix for all artifacts
    aws_access_key_id: null # AWS access key (uses env var if null)
    aws_secret_access_key: null # AWS secret key (uses env var if null)
    region_name: us-east-1 # AWS region
```

**Environment Variables**:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region

**IAM Permissions Required**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": ["arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"]
    }
  ]
}
```

### Google Cloud Storage

```yaml
artifact_storage:
  backend: gcs
  gcs:
    bucket: my-bucket # GCS bucket name (required)
    prefix: models # Optional prefix for all artifacts
    project_id: my-project # GCP project ID (uses env var if null)
    credentials_path: null # Path to service account JSON (uses env var if null)
```

**Environment Variables**:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON file
- `GOOGLE_CLOUD_PROJECT`: GCP project ID

**Service Account Permissions Required**:

```json
{
  "version": 1,
  "etag": "BmRWOtYP8vYF",
  "bindings": [
    {
      "role": "roles/storage.objectAdmin",
      "members": ["serviceAccount:my-sa@my-project.iam.gserviceaccount.com"]
    }
  ]
}
```

## Usage

### In Training Scripts

```python
from astroml.storage import create_artifact_store
from astroml.tracking import MLflowTracker

# Create artifact store from URI
artifact_store = create_artifact_store("s3://my-bucket/models")

# Initialize tracker with artifact store
tracker = MLflowTracker(
    enabled=True,
    artifact_store=artifact_store
)

# Log model - saves to both MLflow and artifact store
artifact_uri = tracker.log_model_artifact(
    model=model,
    artifact_path="model",
    checkpoint_path="best_model.pth"
)
print(f"Model saved to: {artifact_uri}")

# Save arbitrary artifacts
config_uri = tracker.save_artifact(
    local_path="config.yaml",
    artifact_path="config"
)

# Load artifacts back
tracker.load_artifact(
    remote_path="model/best_model.pth",
    local_path="downloaded_model.pth"
)
```

### With Hydra Configuration

```python
from hydra import compose, initialize_config_dir
from astroml.storage import create_artifact_store
from astroml.tracking import MLflowTracker

# Load config with artifact storage settings
cfg = compose(config_name="config", overrides=[
    "artifact_storage=s3"  # Use S3 backend
])

# Create artifact store from config
artifact_uri = cfg.training.artifact_storage.get_artifact_uri()
artifact_store = create_artifact_store(artifact_uri)

# Use with tracker
tracker = MLflowTracker(artifact_store=artifact_store)
```

### Direct Artifact Store Usage

```python
from astroml.storage import create_artifact_store

# Create store
store = create_artifact_store("s3://my-bucket/models")

# Save artifact
uri = store.save("local_model.pth", "experiments/exp1/model.pth")
print(f"Saved to: {uri}")

# Check if exists
if store.exists("experiments/exp1/model.pth"):
    # Load artifact
    store.load("experiments/exp1/model.pth", "downloaded_model.pth")

# List artifacts
artifacts = store.list_artifacts("experiments/exp1")
for artifact in artifacts:
    print(artifact)

# Delete artifact
store.delete("experiments/exp1/model.pth")
```

## URI Format

Artifact URIs follow a standard format:

- **Local**: `file:///path/to/artifacts`
- **S3**: `s3://bucket-name/prefix`
- **GCS**: `gs://bucket-name/prefix`

## API Reference

### ArtifactStore (Abstract Base Class)

```python
class ArtifactStore(ABC):
    def save(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Save local file to artifact store. Returns artifact URI."""

    def load(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Load artifact from store to local filesystem."""

    def exists(self, remote_path: str) -> bool:
        """Check if artifact exists."""

    def delete(self, remote_path: str) -> None:
        """Delete artifact from store."""

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifacts in store."""

    def get_uri(self, remote_path: str) -> str:
        """Get full URI for artifact."""
```

### Factory Function

```python
def create_artifact_store(artifact_uri: str, **kwargs) -> ArtifactStore:
    """Create artifact store from URI.

    Args:
        artifact_uri: URI specifying storage backend
        **kwargs: Additional arguments for store constructor

    Returns:
        Configured ArtifactStore instance
    """
```

### MLflowTracker Integration

```python
class MLflowTracker:
    def __init__(
        self,
        enabled: bool = True,
        tracking_uri: str = "mlruns",
        experiment_name: str = "astroml_experiment",
        run_name: Optional[str] = None,
        log_model_weights: bool = True,
        artifact_uri: Optional[str] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ):
        """Initialize tracker with optional artifact store."""

    def log_model_artifact(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        checkpoint_path: Optional[str] = None,
    ) -> Optional[str]:
        """Log model to MLflow and artifact store. Returns artifact URI."""

    def save_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: str = "artifacts",
    ) -> Optional[str]:
        """Save arbitrary artifact to MLflow and artifact store."""

    def load_artifact(
        self,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> Path:
        """Load artifact from artifact store."""
```

## Examples

### Example 1: Local Development

```python
from astroml.storage import LocalArtifactStore
from astroml.tracking import MLflowTracker

# Use local storage for development
store = LocalArtifactStore("./artifacts")
tracker = MLflowTracker(artifact_store=store)

# Train and save
tracker.log_model_artifact(model, checkpoint_path="best.pth")
```

### Example 2: Production with S3

```python
from astroml.storage import S3ArtifactStore
from astroml.tracking import MLflowTracker

# Use S3 for production
store = S3ArtifactStore(
    bucket="prod-models",
    prefix="astroml",
    region_name="us-west-2"
)
tracker = MLflowTracker(artifact_store=store)

# Train and save
uri = tracker.log_model_artifact(model, checkpoint_path="best.pth")
print(f"Model available at: {uri}")
```

### Example 3: Multi-Cloud with GCS

```python
from astroml.storage import GCSArtifactStore
from astroml.tracking import MLflowTracker

# Use GCS for multi-cloud setup
store = GCSArtifactStore(
    bucket="ml-artifacts",
    prefix="astroml-experiments",
    project_id="my-gcp-project"
)
tracker = MLflowTracker(artifact_store=store)

# Train and save
uri = tracker.log_model_artifact(model, checkpoint_path="best.pth")
```

## Troubleshooting

### S3 Connection Issues

**Problem**: `NoCredentialsError` when connecting to S3

**Solution**: Ensure AWS credentials are configured:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

Or configure in `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your_key
aws_secret_access_key = your_secret
```

### GCS Connection Issues

**Problem**: `google.auth.exceptions.DefaultCredentialsError` when connecting to GCS

**Solution**: Set up service account credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=my-project
```

### Permission Denied Errors

**Problem**: `PermissionError` when saving to S3/GCS

**Solution**: Verify IAM permissions for your credentials. Ensure the principal has:

- `s3:PutObject` (S3) or `storage.objects.create` (GCS)
- `s3:GetObject` (S3) or `storage.objects.get` (GCS)
- `s3:DeleteObject` (S3) or `storage.objects.delete` (GCS)

## Performance Considerations

- **Local Storage**: Fastest for local development, no network overhead
- **S3**: Good for AWS environments, supports multipart uploads for large files
- **GCS**: Good for GCP environments, similar performance to S3

For large models (>1GB), consider:

- Using multipart uploads (handled automatically by fsspec)
- Compressing models before upload
- Using regional buckets for faster access

## Migration Guide

### From Local to S3

```python
# Old: Local storage only
torch.save(model.state_dict(), "outputs/model.pth")

# New: S3 storage
from astroml.storage import S3ArtifactStore

store = S3ArtifactStore("my-bucket", "models")
store.save("outputs/model.pth", "experiment1/model.pth")
```

### From Direct Saves to Artifact Store

```python
# Old: Direct file operations
import shutil
shutil.copy("model.pth", "outputs/model.pth")

# New: Artifact store
from astroml.storage import create_artifact_store

store = create_artifact_store("s3://my-bucket/models")
store.save("model.pth", "experiment1/model.pth")
```

## Dependencies

The artifact storage system requires:

- `fsspec`: Filesystem abstraction layer
- `s3fs`: S3 support (for S3 backend)
- `gcsfs`: GCS support (for GCS backend)

Install with:

```bash
pip install fsspec s3fs gcsfs
```

Or install with specific backends:

```bash
pip install fsspec[s3]  # S3 only
pip install fsspec[gcs]  # GCS only
pip install fsspec[s3,gcs]  # Both
```
