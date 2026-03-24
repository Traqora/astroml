# Docker Deployment Guide for AstroML

This guide provides comprehensive instructions for deploying AstroML using Docker and Docker Compose.

## 🐳 Overview

The AstroML Docker setup includes:

- **Multi-stage Dockerfile** with optimized images for different use cases
- **Docker Compose** configuration for complete environment setup
- **GPU support** for ML training
- **Development**, **production**, and **monitoring** profiles

## 🏗 Docker Build Stages

### Available Build Targets

| Stage | Purpose | Base Image | Use Case |
|-------|---------|------------|----------|
| `base` | Common dependencies | `python:3.11-slim` | Foundation for all stages |
| `ingestion` | Data ingestion & streaming | `base` | Stellar data ingestion |
| `training` | GPU-enabled ML training | `nvidia/cuda:12.1-runtime` | GPU training environments |
| `training-cpu` | CPU-only ML training | `base` | CPU training environments |
| `development` | Development with tools | `base` | Local development |
| `production` | Minimal production image | `base` | Production deployment |

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU support)

### Basic Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone https://github.com/tecch-wiz/astroml.git
   cd astroml
   ```

2. **Start the basic environment:**
   ```bash
   docker-compose up -d postgres redis
   ```

3. **Run database migrations:**
   ```bash
   docker-compose run --rm ingestion python -m alembic upgrade head
   ```

4. **Start ingestion service:**
   ```bash
   docker-compose up -d ingestion
   ```

## 📋 Docker Compose Services

### Core Services

- **postgres**: PostgreSQL database with health checks
- **redis**: Redis for caching and job queues
- **ingestion**: Main data ingestion service
- **streaming**: Enhanced streaming service

### Training Services

- **training-gpu**: GPU-enabled training (requires NVIDIA Docker)
- **training-cpu**: CPU-only training

### Optional Services

- **dev**: Development environment with Jupyter
- **production**: Production-optimized service
- **prometheus**: Monitoring and metrics
- **grafana**: Visualization dashboard

## 🛠 Usage Examples

### Development Environment

```bash
# Start development services
docker-compose --profile dev up -d

# Access Jupyter Lab
open http://localhost:8888
```

### GPU Training

```bash
# Start GPU training service
docker-compose --profile gpu up -d training-gpu

# Run training
docker-compose exec training-gpu python -m astroml.training.train_gcn
```

### CPU Training

```bash
# Start CPU training service
docker-compose --profile cpu up -d training-cpu

# Run training
docker-compose exec training-cpu python -m astroml.training.train_gcn
```

### Production Deployment

```bash
# Deploy production services
docker-compose --profile prod up -d production
```

### Monitoring Stack

```bash
# Start monitoring services
docker-compose --profile monitoring up -d prometheus grafana

# Access Grafana
open http://localhost:3000  # admin/admin
```

## 🔧 Configuration

### Environment Variables

Key environment variables for services:

```yaml
# Database
DATABASE_URL: postgresql://astroml:astroml_password@postgres:5432/astroml

# Redis
REDIS_URL: redis://redis:6379/0

# Stellar Network
STELLAR_NETWORK_PASSPHRASE: "Public Global Stellar Network ; September 2015"
STELLAR_HORIZON_URL: https://horizon.stellar.org

# Logging
LOG_LEVEL: INFO
```

### Custom Configuration

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  ingestion:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./local_config:/app/config:ro

  postgres:
    ports:
      - "5433:5432"  # Different port to avoid conflicts
```

## 📊 GPU Support

### NVIDIA Docker Setup

1. **Install NVIDIA Container Toolkit:**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Test GPU support:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
   ```

### GPU Training

```bash
# Build GPU image
docker build --target training -t astroml:training-gpu .

# Run with GPU
docker run --gpus all -v $(pwd):/app astroml:training-gpu python -m astroml.training.train_gcn
```

## 🔍 Monitoring and Logging

### Log Access

```bash
# View ingestion logs
docker-compose logs -f ingestion

# View all service logs
docker-compose logs -f

# View specific number of lines
docker-compose logs --tail=100 training-gpu
```

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# Detailed health status
curl http://localhost:8000/health  # If health endpoint is exposed
```

### Monitoring Stack

With the monitoring profile enabled:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🗂 Data Persistence

### Volume Structure

```
volumes/
├── postgres_data/          # PostgreSQL data
├── redis_data/             # Redis data
├── ingestion_logs/         # Ingestion logs
├── ingestion_data/         # Ingestion data
├── training_models/        # Trained models
├── training_data/          # Training datasets
├── training_logs/          # Training logs
└── production_data/        # Production data
```

### Backup and Restore

```bash
# Backup database
docker-compose exec postgres pg_dump -U astroml astroml > backup.sql

# Restore database
docker-compose exec -T postgres psql -U astroml astroml < backup.sql

# Backup volumes
docker run --rm -v astroml_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz -C /data .
```

## 🧪 Testing

### Running Tests in Docker

```bash
# Run all tests
docker-compose run --rm dev python -m pytest tests/ -v

# Run with coverage
docker-compose run --rm dev python -m pytest tests/ --cov=astroml --cov-report=html

# Run specific test file
docker-compose run --rm dev python -m pytest tests/test_structural_importance.py -v
```

### Integration Testing

```bash
# Test ingestion pipeline
docker-compose run --rm ingestion python -c "import astroml.ingestion; print('OK')"

# Test training environment
docker-compose run --rm training-cpu python -c "import torch; import torch_geometric; print('OK')"
```

## 🚀 Production Deployment

### Production Checklist

- [ ] Use `production` build target
- [ ] Configure proper secrets management
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Set up backup strategy
- [ ] Review resource limits
- [ ] Test disaster recovery

### Production Commands

```bash
# Build production image
docker build --target production -t astroml:prod .

# Deploy with production profile
docker-compose --profile prod up -d production

# Scale services
docker-compose --profile prod up -d --scale production=3
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU not detected:**
   ```bash
   # Check NVIDIA Docker installation
   docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
   ```

2. **Database connection issues:**
   ```bash
   # Check database health
   docker-compose exec postgres pg_isready -U astroml
   ```

3. **Permission issues:**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER .dockerignore
   ```

4. **Out of memory:**
   ```bash
   # Check resource usage
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

### Debug Mode

```bash
# Run with shell access
docker-compose run --rm ingestion bash

# Debug with environment variables
docker-compose run --rm -e DEBUG=1 ingestion python -m astroml.ingestion
```

## 📚 Advanced Usage

### Custom Images

```bash
# Build custom stage
docker build --target development -t astroml:dev .

# Build with custom arguments
docker build --build-arg PYTHON_VERSION=3.10 -t astroml:custom .
```

### Multi-Node Deployment

```bash
# Initialize Docker Swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml astroml
```

### Performance Tuning

```yaml
# docker-compose.yml performance tweaks
services:
  training-gpu:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🆘 Support

For Docker-related issues:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review service logs: `docker-compose logs <service>`
3. Verify resource usage: `docker stats`
4. Test with minimal configuration

For application issues, refer to the main AstroML documentation.
