# Artifact Store Integration Guide

This guide explains how to integrate the new artifact storage system into your existing AstroML workflows.

## What Changed

### New Modules

1. **`astroml/storage/artifact_store.py`** - Core artifact store implementations
   - `ArtifactStore` - Abstract base class
   - `LocalArtifactStore` - Local filesystem storage
   - `S3ArtifactStore` - AWS S3 storage
   - `GCSArtifactStore` - Google Cloud Storage
   - `create_artifact_store()` - Factory function

2. **`astroml/storage/config.py`** - Configuration classes
   - `ArtifactStorageConfig` - Main configuration
   - `LocalStorageConfig` - Local storage settings
   - `S3StorageConfig` - S3 settings
   - `GCSStorageConfig` - GCS settings

3. **`astroml/storage/__init__.py`** - Module exports

### Updated Modules

1. **`astroml/tracking/mlflow_tracker.py`** - Enhanced with artifact store support
   - New parameters: `artifact_uri`, `artifact_store`
   - New methods: `save_artifact()`, `load_artifact()`
   - `log_model_artifact()` now returns artifact URI

2. **`astroml/training/config.py`** - Added artifact storage configuration
   - New field: `artifact_storage: ArtifactStorageConfig`

### New Configuration Files

- `configs/artifact_storage/local.yaml` - Local storage config
- `configs/artifact_storage/s3.yaml` - S3 storage config
- `configs/artifact_storage/gcs.yaml` - GCS storage config

### New Dependencies

Added to `requirements.txt` and `requirements-cpu.txt`:

- `fsspec>=2024.2.0` - Filesystem abstraction
- `s3fs>=2024.2.0` - S3 support
- `gcsfs>=2024.2.0` - GCS support

## Migration Path

### Step 1: Update Dependencies

```bash
pip install -r requirements.txt  # or requirements-cpu.txt
```

### Step 2: Update Training Scripts

**Before:**

```python
from astroml.tracking import MLflowTracker

tracker = MLflowTracker(
    enabled=True,
    tracking_uri="mlruns",
    experiment_name="my_experiment"
)

# Models saved only to local filesystem
torch.save(model.state_dict(), "outputs/model.pth")
tracker.log_model_artifact(model, checkpoint_path="outputs/model.pth")
```

**After:**

```python
from astroml.storage import create_artifact_store
from astroml.tracking import MLflowTracker

# Create artifact store
artifact_store = create_artifact_store("s3://my-bucket/models")

# Initialize tracker with artifact store
tracker = MLflowTracker(
    enabled=True,
    tracking_uri="mlruns",
    experiment_name="my_experiment",
    artifact_store=artifact_store
)

# Models saved to both MLflow and S3
torch.save(model.state_dict(), "outputs/model.pth")
artifact_uri = tracker.log_model_artifact(
    model,
    checkpoint_path="outputs/model.pth"
)
print(f"Model saved to: {artifact_uri}")
```

### Step 3: Update Hydra Configuration

**Before:**

```yaml
# configs/config.yaml
experiment:
  name: "astroml_experiment"
  save_dir: "outputs"

mlflow:
  enabled: true
  tracking_uri: "mlruns"
```

**After:**

```yaml
# configs/config.yaml
defaults:
  - artifact_storage: local # or s3, gcs

experiment:
  name: "astroml_experiment"
  save_dir: "outputs"

mlflow:
  enabled: true
  tracking_uri: "mlruns"
```

Then create `configs/artifact_storage/local.yaml`:

```yaml
artifact_storage:
  backend: local
  local:
    path: artifacts
```

### Step 4: Use in Training

```python
from hydra import compose, initialize_config_dir
from astroml.storage import create_artifact_store
from astroml.training.config import TrainingConfig

# Load config
cfg = compose(config_name="config")

# Get artifact URI from config
artifact_uri = cfg.training.artifact_storage.get_artifact_uri()

# Create store
artifact_store = create_artifact_store(artifact_uri)

# Use with tracker
tracker = MLflowTracker(artifact_store=artifact_store)
```

## Common Patterns

### Pattern 1: Local Development, S3 Production

```python
import os
from astroml.storage import create_artifact_store

# Use environment variable to switch backends
artifact_uri = os.getenv(
    "ARTIFACT_URI",
    "file:///tmp/artifacts"  # Default to local
)

artifact_store = create_artifact_store(artifact_uri)
```

**Usage:**

```bash
# Development
python train.py

# Production
export ARTIFACT_URI="s3://prod-models/astroml"
python train.py
```

### Pattern 2: Multi-Experiment Tracking

```python
from astroml.storage import S3ArtifactStore

# Each experiment gets its own prefix
experiment_id = "exp_2024_05_31_001"
store = S3ArtifactStore(
    bucket="ml-experiments",
    prefix=f"astroml/{experiment_id}"
)

tracker = MLflowTracker(artifact_store=store)
```

### Pattern 3: Artifact Versioning

```python
from pathlib import Path
from astroml.storage import create_artifact_store

store = create_artifact_store("s3://models/astroml")

# Save with version
version = "v1.0.0"
model_path = f"models/{version}/best_model.pth"
uri = store.save("best_model.pth", model_path)

# Later, load specific version
store.load(f"models/v1.0.0/best_model.pth", "model_v1.pth")
store.load(f"models/v1.1.0/best_model.pth", "model_v1_1.pth")
```

### Pattern 4: Artifact Cleanup

```python
from astroml.storage import create_artifact_store

store = create_artifact_store("s3://models/astroml")

# List and delete old artifacts
artifacts = store.list_artifacts("experiments/old")
for artifact in artifacts:
    store.delete(artifact)
    print(f"Deleted: {artifact}")
```

## Backward Compatibility

The changes are **fully backward compatible**:

1. **Existing code without artifact store still works**

   ```python
   # This still works - no artifact store
   tracker = MLflowTracker(enabled=True)
   tracker.log_model_artifact(model, checkpoint_path="model.pth")
   ```

2. **Existing config files still work**

   ```yaml
   # Old config without artifact_storage still works
   experiment:
     save_dir: "outputs"
   ```

3. **MLflow tracking unchanged**
   - Models still logged to MLflow as before
   - Artifact store is optional enhancement

## Testing

Run the test suite:

```bash
# Run all artifact store tests
pytest tests/test_artifact_store.py -v

# Run specific test
pytest tests/test_artifact_store.py::TestLocalArtifactStore::test_save_and_load -v

# Run with coverage
pytest tests/test_artifact_store.py --cov=astroml.storage
```

## Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'fsspec'`

**Solution:** Install dependencies

```bash
pip install -r requirements.txt
```

### S3 Connection Error: `NoCredentialsError`

**Solution:** Configure AWS credentials

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### GCS Connection Error: `DefaultCredentialsError`

**Solution:** Configure GCP credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=my-project
```

### Permission Denied When Saving

**Solution:** Verify IAM permissions for your credentials

For S3:

- `s3:PutObject`
- `s3:GetObject`
- `s3:DeleteObject`
- `s3:ListBucket`

For GCS:

- `storage.objects.create`
- `storage.objects.get`
- `storage.objects.delete`
- `storage.buckets.get`

## Performance Tips

1. **Use regional buckets** for faster access
2. **Compress large models** before upload
3. **Use multipart uploads** for files >100MB (automatic with fsspec)
4. **Cache frequently accessed artifacts** locally
5. **Use prefixes** to organize artifacts logically

## Next Steps

1. **Update your training scripts** to use artifact stores
2. **Configure your preferred backend** (local, S3, or GCS)
3. **Test with sample models** before production use
4. **Monitor artifact storage costs** if using cloud backends
5. **Set up artifact cleanup policies** for old experiments

## Additional Resources

- [ARTIFACT_STORAGE.md](ARTIFACT_STORAGE.md) - Detailed configuration guide
- [examples/train_with_artifact_store.py](examples/train_with_artifact_store.py) - Example training script
- [fsspec documentation](https://filesystem_spec.readthedocs.io/)
- [s3fs documentation](https://s3fs.readthedocs.io/)
- [gcsfs documentation](https://gcsfs.readthedocs.io/)
