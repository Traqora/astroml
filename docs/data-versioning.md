# Data Versioning with DVC

AstroML integrates DVC (Data Version Control) for dataset versioning,
pipeline reproducibility, and collaboration on data science projects.

## Overview

DVC tracks large data files, ML pipelines, and experiments with
versioned metadata while keeping data storage efficient through
content-addressable storage and caching.

### Key Features

- **Dataset versioning** – Track and version any dataset, single file or directory.
- **Pipeline reproducibility** – Define reproducible ML pipelines as DVC stages.
- **Remote storage** – Push data to S3, GCS, or local remotes.
- **Version comparison** – Diff between dataset versions to see file changes.
- **Tagging & annotation** – Tag versions (e.g., `production`, `staging`) and add annotations.
- **CI/CD integration** – Run data pipelines in CI with automatic caching.

## Setup

### 1. Install DVC

```bash
pip install dvc[s3]   # For S3 remote
pip install dvc[gs]   # For GCS remote
pip install dvc        # For local-only
```

### 2. Configure the Remote

Edit `.dvc/config` to point to your storage backend:

```ini
[core]
    remote = astroml-remote
    autostage = true

['remote "astroml-remote"']
    url = s3://your-bucket/dvc-store
    endpointurl = https://s3.amazonaws.com
```

### 3. Verify Setup

```python
from astroml.storage.data_versioning import DataVersionControl

dvc = DataVersionControl()
print("DVC enabled:", dvc.enabled)
```

## Usage

### Adding a Dataset

```python
from astroml.storage.data_versioning import DataVersionControl

dvc = DataVersionControl()

# Add and version a dataset
ver = dvc.add_dataset(
    name="training_data",
    path="data/train.csv",
    version="v1.0.0",
    description="Initial training dataset",
    tags=["production", "v1"],
)

print(f"Version: {ver.version_id}")
print(f"DVC hash: {ver.dvc_hash}")
```

### Tagging and Annotation

```python
# Add tags
dvc.tag_version(ver.version_id, ["reviewed", "validated"])

# Add annotations
dvc.annotate(ver.version_id, {
    "source": "data warehouse",
    "rows": "50000",
    "validation_score": "0.95",
})
```

### Comparing Versions

```python
diff = dvc.compare_versions("version-id-1", "version-id-2")
print(diff.summary)
# Output: "Comparing v1.0.0 → v1.1.0; 3 added; 1 removed; 2 modified; Size diff: +1,024,345 bytes"
```

### Push/Pull to Remote

```python
# Push all tracked data
dvc.push()

# Pull from remote
dvc.pull()
```

## Pipeline Reproducibility

Use `DVCPipelineManager` to define and run reproducible pipelines:

```python
from astroml.pipeline.dvc_pipeline import DVCPipelineManager

mgr = DVCPipelineManager()

# Create a pipeline
pipe = mgr.create_pipeline(
    name="feature-engineering",
    description="Feature engineering pipeline",
)

# Add stages
mgr.add_stage(
    pipe.pipeline_id,
    name="clean-data",
    command="python scripts/clean.py --input data/raw.csv --output data/clean.csv",
    dependencies=["data/raw.csv", "scripts/clean.py"],
    outputs=["data/clean.csv"],
)

mgr.add_stage(
    pipe.pipeline_id,
    name="extract-features",
    command="python scripts/features.py --input data/clean.csv --output data/features.csv",
    dependencies=["data/clean.csv", "scripts/features.py"],
    outputs=["data/features.csv"],
)

# Run the pipeline
run = mgr.run(pipe.pipeline_id)
print(f"Status: {run.status}")

# Check runs
runs = mgr.list_runs(pipe.pipeline_id)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/data-versioning/datasets` | Add a dataset |
| GET | `/api/v1/data-versioning/datasets` | List datasets |
| GET | `/api/v1/data-versioning/datasets/{id}` | Get dataset version |
| POST | `/api/v1/data-versioning/datasets/{id}/tags` | Add tags |
| POST | `/api/v1/data-versioning/datasets/{id}/annotations` | Add annotations |
| GET | `/api/v1/data-versioning/compare` | Compare two versions |
| POST | `/api/v1/data-versioning/push` | Push to remote |
| POST | `/api/v1/data-versioning/pull` | Pull from remote |
| GET | `/api/v1/data-versioning/status` | DVC repository status |
| GET | `/api/v1/data-versioning/datasets/{id}/snapshot` | Get version snapshot |
| GET | `/api/v1/data-versioning/datasets/{id}/export` | Export version |

## CI/CD Integration

The DVC pipeline integrates with GitHub Actions via `.dvc/config` and the
`dvc_pipeline.py` module. Add a workflow step:

```yaml
- name: Run DVC pipeline
  run: |
    python -c "
    from astroml.pipeline.dvc_pipeline import DVCPipelineManager
    mgr = DVCPipelineManager()
    run = mgr.run('feature-engineering')
    assert run.status == 'completed', f'Pipeline failed: {run.error}'
    "
```

## Troubleshooting

- **DVC not detected**: Ensure `dvc` is installed and a `.dvc/` directory exists.
- **Push fails**: Verify remote configuration in `.dvc/config` and AWS/GCP credentials.
- **Cache issues**: Run `dvc gc --workspace` to clean up unused cache entries.
- **Lock conflicts**: Stage files are content-addressed; conflicts are rare but can occur with concurrent writes.

## See Also

- [DVC Documentation](https://dvc.org/doc)
- [Configuration Reference](CONFIGURATION.md)
- [GitOps Workflow](gitops-workflow.md)