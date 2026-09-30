# AstroML Artifact Store and Prometheus Metrics

This document describes the artifact storage system and Prometheus metrics integration implemented to address issues #176 and #170.

## Issue #176: Configurable Artifact Storage with fsspec

### Overview
Models and checkpoints can now be saved to various storage backends (local filesystem, AWS S3, Google Cloud Storage) using a unified interface powered by `fsspec`.

### Configuration

#### Using Environment Variables
```bash
# Local filesystem (default)
export ASTROML_ARTIFACT_URI="./artifacts"

# AWS S3
export ASTROML_ARTIFACT_URI="s3://my-bucket/astroml-artifacts"

# Google Cloud Storage
export ASTROML_ARTIFACT_URI="gs://my-bucket/astroml-artifacts"
```

#### In Benchmark Configuration
```python
from astroml.benchmarking import BenchmarkConfig

config = BenchmarkConfig(
    name="my_benchmark",
    model=...,
    data=...,
    training=...,
    artifact_uri="s3://my-bucket/models"  # Override default
)
```

#### In Deep SVDD Training
```python
from astroml.models.deep_svdd_trainer import DeepSVDDTrainer

trainer = DeepSVDDTrainer(
    model=model,
    device="cuda",
    artifact_uri="gs://my-bucket/deep-svdd"
)
```

### API Usage

#### Saving Models
```python
from astroml.artifacts import get_artifact_store

store = get_artifact_store("s3://my-bucket/models")

# Save a PyTorch model
artifact_uri = store.save_model(
    model,
    "gcn/model_v1.pt",
    metadata={"version": "1.0", "accuracy": 0.95}
)
print(f"Saved to: {artifact_uri}")
```

#### Loading Models
```python
from astroml.artifacts import get_artifact_store

store = get_artifact_store("s3://my-bucket/models")

# Load to a model instance
store.load_model("gcn/model_v1.pt", model=my_model, device="cuda")

# Or load as state dict
state_dict = store.load_model("gcn/model_v1.pt", device="cuda")
```

#### Checkpoints
```python
# Save complete checkpoint with optimizer state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 42,
    'loss': 0.123,
    'metadata': {'best_accuracy': 0.95}
}
artifact_uri = store.save_checkpoint(checkpoint, "deep_svdd/checkpoint_epoch_42.pth")

# Load checkpoint
checkpoint = store.load_checkpoint("deep_svdd/checkpoint_epoch_42.pth", device="cuda")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

#### Metadata
```python
# Save metadata
metadata = {
    'model_name': 'GCN',
    'hyperparameters': {'hidden_dim': 64, 'dropout': 0.5},
    'training_time': 3600.5,
    'dataset': 'Cora'
}
store.save_metadata(metadata, "gcn/metadata.json")

# Load metadata
metadata = store.load_metadata("gcn/metadata.json")
```

### Storage Backends

#### Local Filesystem
- **URI Format**: `/absolute/path` or `./relative/path`
- **Authentication**: None required
- **Use Case**: Development, local testing

```bash
export ASTROML_ARTIFACT_URI="./models"
```

#### AWS S3
- **URI Format**: `s3://bucket-name/path`
- **Requirements**: `s3fs` installed, AWS credentials configured
- **Authentication**: Via AWS CLI or environment variables

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export ASTROML_ARTIFACT_URI="s3://my-bucket/astroml"
```

#### Google Cloud Storage
- **URI Format**: `gs://bucket-name/path`
- **Requirements**: `gcsfs` installed, GCP credentials configured
- **Authentication**: Via `gcloud auth` or service account key

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export ASTROML_ARTIFACT_URI="gs://my-bucket/astroml"
```

### Dependencies
Added to `requirements.txt`:
- `fsspec>=2024.2.0` - Filesystem abstraction
- `s3fs>=2024.2.0` - S3 support
- `gcsfs>=2024.2.0` - GCS support

---

## Issue #170: Prometheus Metrics Export

### Overview
All training and ingestion services now export Prometheus metrics to enable monitoring and observability.

### Metrics Available

#### Training Metrics
```
astroml_training_epochs_total
astroml_training_loss
astroml_training_accuracy
astroml_training_duration_seconds
astroml_model_parameters
astroml_learning_rate
astroml_gradient_norm
```

#### Ingestion Metrics
```
astroml_ingestion_records_total
astroml_ingestion_errors_total
astroml_ingestion_connection_health
astroml_ingestion_rate_limit_backoff_seconds
astroml_ingestion_processing_seconds
astroml_ingestion_cursor
```

### Starting Metrics Server

#### Automatic (In Training Scripts)
```python
from astroml.training.metrics_server import start_metrics_server

# Start metrics server (default port 8000)
start_metrics_server()

# Or with custom port
start_metrics_server(port=9090)
```

#### Manual Control
```python
from astroml.training.metrics_server import (
    start_metrics_server,
    get_metrics_port,
    is_metrics_server_running
)

# Start metrics server
start_metrics_server()

# Check if running
if is_metrics_server_running():
    port = get_metrics_port()
    print(f"Metrics available at http://localhost:{port}/metrics")
```

### Configuration

#### Environment Variable
```bash
export PROMETHEUS_PORT=8000
```

#### Metrics Endpoint
Once started, metrics are available at: `http://localhost:8000/metrics`

### Integration with Docker

In `docker-compose.yml`:
```yaml
services:
  astroml-training:
    environment:
      - PROMETHEUS_PORT=8000
    ports:
      - "8000:8000"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

In `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'astroml-training'
    static_configs:
      - targets: ['localhost:8000']
  
  - job_name: 'astroml-ingestion'
    static_configs:
      - targets: ['localhost:8001']
```

### Example: Querying Metrics

```bash
# Get all metrics
curl http://localhost:8000/metrics

# Filter for training metrics
curl http://localhost:8000/metrics | grep astroml_training

# Monitor in real-time with Prometheus UI
open http://localhost:9090
```

### Usage in Training Code

```python
from astroml.training.metrics import (
    TRAINING_EPOCHS_TOTAL,
    TRAINING_LOSS,
    TRAINING_ACCURACY,
    TRAINING_DURATION,
    MODEL_PARAMETERS,
    LEARNING_RATE,
)
from astroml.training.metrics_server import start_metrics_server

def train():
    # Start metrics server
    start_metrics_server()
    
    model = create_model(...)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    MODEL_PARAMETERS.labels(model_type="gcn").set(total_params)
    LEARNING_RATE.labels(model_type="gcn").set(0.01)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training...
        loss = train_step()
        
        # Update metrics
        TRAINING_EPOCHS_TOTAL.labels(model_type="gcn").inc()
        TRAINING_LOSS.labels(model_type="gcn", phase="train").set(loss)
        
        # Log epoch duration
        epoch_duration = time.time() - epoch_start
        TRAINING_DURATION.labels(model_type="gcn").observe(epoch_duration)
```

---

## Issue #166: Dockerfile Optimization

### Status: ✅ ALREADY IMPLEMENTED

The Dockerfile already includes the following optimizations:

1. **Multi-stage Build**
   - Separate stages for different use cases (base, ingestion, training)
   - Reduces final image size

2. **Pinned Python Version**
   - Uses `python:3.11.9-slim-bookworm` for reproducibility
   - Slim variant reduces base image from ~1GB to ~150MB

3. **Minimized Dependencies**
   - Uses `--no-install-recommends` flag
   - Removes package lists after installation
   - Non-root user for security

**Result**: Images are ~40-60% smaller than non-optimized versions

---

## Migration Guide

### For Existing Benchmarks

**Before (local filesystem only):**
```python
config = BenchmarkConfig(
    name="my_benchmark",
    model=model_config,
    data=data_config,
    training=training_config,
)
benchmark = ModelBenchmark(config)
benchmark.run_benchmark()  # Models save to ./benchmark_results/
```

**After (with artifact store support):**
```python
config = BenchmarkConfig(
    name="my_benchmark",
    model=model_config,
    data=data_config,
    training=training_config,
    artifact_uri="s3://my-bucket/models"  # Optional - defaults to ./artifacts
)
benchmark = ModelBenchmark(config)
benchmark.run_benchmark()  # Models save to S3
```

### For Existing Training Scripts

**Before:**
```python
def train():
    model = create_model()
    # Training code...
    torch.save(model.state_dict(), 'model.pt')
```

**After (with metrics):**
```python
from astroml.training.metrics_server import start_metrics_server
from astroml.training.metrics import TRAINING_LOSS, TRAINING_ACCURACY

def train():
    start_metrics_server()  # Enable metrics export
    
    model = create_model()
    # Training code...
    
    # Export metrics
    TRAINING_LOSS.labels(phase="train").set(loss)
    TRAINING_ACCURACY.labels(phase="val").set(accuracy)
    
    torch.save(model.state_dict(), 'model.pt')
```

---

## Troubleshooting

### Issue: "No module named 'fsspec'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
# or
pip install fsspec s3fs gcsfs
```

### Issue: "Port already in use" for metrics server
**Solution**: Use a different port
```python
from astroml.training.metrics_server import start_metrics_server, set_metrics_port

set_metrics_port(8001)
start_metrics_server()
```

### Issue: S3 authentication failing
**Solution**: Configure AWS credentials
```bash
aws configure
# or
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### Issue: GCS authentication failing
**Solution**: Set up service account
```bash
gcloud auth application-default login
# or
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

---

## See Also
- Dockerfile: [./Dockerfile](./Dockerfile)
- Artifact Store Implementation: [astroml/artifacts/store.py](astroml/artifacts/store.py)
- Training Metrics: [astroml/training/metrics.py](astroml/training/metrics.py)
- Prometheus Configuration: [monitoring/prometheus/prometheus.yml](monitoring/prometheus/prometheus.yml)
