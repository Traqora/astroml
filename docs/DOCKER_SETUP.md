# Docker Setup Guide for AstroML

## Overview

This guide provides comprehensive instructions for setting up, developing, training, testing, and deploying AstroML using Docker. It combines containerized development, PostgreSQL, Redis, Feature Store services, GPU-enabled training, monitoring, and production deployment into a single Docker workflow.

## Table of Contents

1. Prerequisites
2. Quick Start
3. Docker Services
4. Docker Build Stages
5. Environment Configuration
6. Development Workflow
7. Common Operations
8. Production Deployment
9. Troubleshooting
10. Advanced Usage
11. Security Best Practices

---

## Prerequisites

### System Requirements

- Docker Engine 20.10+
- Docker Compose v2+
- 8GB+ RAM (development)
- 16GB+ RAM (training workloads)
- NVIDIA GPU (optional for GPU training)
- 20GB+ available disk space

### Docker Installation

#### Linux

```

## Quick Start

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Menjay7/astroml.git
cd astroml

cp .env.example .env

# Linux/macOS
chmod +x scripts/docker-dev.sh
```

### 2. Start Core Infrastructure

For local development with native Python execution:

```bash
# Start PostgreSQL and Redis only
docker compose up -d postgres redis

# Verify services
docker compose ps

# Run migrations locally
alembic upgrade head

# Run application locally
python examples/quick_start.py
```

### 3. Start Full Containerized Development Environment

If you prefer to run everything inside Docker:

```bash
# Build images
./scripts/docker-dev.sh build

# Start development environment
./scripts/docker-dev.sh dev

# Or using Docker Compose directly
docker compose --profile dev up -d
```

### 4. Start Application Services

```bash
# Start ingestion service
docker compose up -d ingestion

# Start streaming service
docker compose up -d streaming

# Verify running services
docker compose ps
```

### 5. Start Training

#### CPU Training

```bash
docker compose --profile cpu up training-cpu
```

#### GPU Training

```bash
docker compose --profile gpu up training-gpu
```

Requires NVIDIA Docker runtime and compatible GPU drivers.

### 6. Start Monitoring

```bash
docker compose --profile monitoring up -d
```

Available services:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### 7. Access Services

| Service | URL/Port |
|----------|-----------|
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |
| Feature Store | http://localhost:8000 |
| Ingestion API | http://localhost:8001 |
| Streaming API | http://localhost:8002 |
| Jupyter Lab | http://localhost:8888 |
| TensorBoard (GPU) | http://localhost:6006 |
| TensorBoard (CPU) | http://localhost:6007 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## Docker Services

### Core Infrastructure

#### PostgreSQL Database

- **Container**: `astroml-postgres`
- **Image**: `postgres:15-alpine`
- **Port**: `5432`
- **Database**: `astroml`
- **User**: `astroml`
- **Storage**: Persistent Docker volume (`postgres_data`)
- **Purpose**: Primary application database

#### Redis Cache

- **Container**: `astroml-redis`
- **Image**: `redis:7-alpine`
- **Port**: `6379`
- **Storage**: Persistent Docker volume (`redis_data`)
- **Features**:
  - AOF persistence
  - Job queues
  - Application caching
  - Session storage

#### Feature Store

- **Container**: `astroml-feature-store`
- **Port**: `8000`
- **Storage Path**: `/app/feature_store`
- **Purpose**:
  - Feature management
  - Feature caching
  - Feature versioning
  - ML feature serving

### Application Services

#### Ingestion Service

- **Container**: `astroml-ingestion`
- **Port**: `8001`
- **Purpose**: Data ingestion and preprocessing
- **Dependencies**: PostgreSQL, Redis

#### Streaming Service

- **Container**: `astroml-streaming`
- **Port**: `8002`
- **Purpose**: Real-time data streaming and event processing

#### Development Environment

- **Container**: `astroml-dev`
- **Ports**:
  - API: `8003`
  - Jupyter Lab: `8888`
  - TensorBoard: `6008`
- **Purpose**:
  - Interactive development
  - Notebook experimentation
  - Testing and debugging

#### Production Service

- **Container**: `astroml-production`
- **Port**: `8004`
- **Purpose**: Production deployment

### Training Services

#### GPU Training

- **Container**: `astroml-training-gpu`
- **TensorBoard Port**: `6006`
- **GPU Required**: Yes
- **Purpose**: Accelerated model training

#### CPU Training

- **Container**: `astroml-training-cpu`
- **TensorBoard Port**: `6007`
- **GPU Required**: No
- **Purpose**: CPU-only training workloads

### Monitoring Services

#### Prometheus

- **Container**: `astroml-prometheus`
- **Port**: `9090`
- **Purpose**: Metrics collection and alerting

#### Grafana

- **Container**: `astroml-grafana`
- **Port**: `3000`
- **Purpose**: Monitoring dashboards and visualization
- **Default Credentials**: `admin / admin`

### Application Services

#### Ingestion Service
### Application Services

#### Ingestion Service

- **Container**: `astroml-ingestion`
- **Service Name**: `ingestion`
- **Port**: `8001` (API) / `8080` (Health Check)
- **Purpose**: Data ingestion, ETL processing, and Stellar data collection
- **Environment Variables**:
  - `DATABASE_URL`
  - `REDIS_URL`
  - `LOG_LEVEL`
- **Volumes**:
  - `ingestion_logs`
  - `ingestion_data`
- **Dependencies**: PostgreSQL, Redis

#### Streaming Service

- **Container**: `astroml-streaming`
- **Service Name**: `streaming`
- **Port**: `8002`
- **Purpose**: Real-time data streaming and event processing
- **Volumes**:
  - `streaming_logs`

#### Development Environment

- **Container**: `astroml-dev`
- **Service Name**: `dev`
- **Ports**:
  - `8003` (API)
  - `8888` (Jupyter Lab)
  - `6008` (TensorBoard)
- **Profile**: `dev`
- **Purpose**:
  - Interactive development
  - Live code editing
  - Testing and debugging
  - Jupyter notebooks

#### Production Service

- **Container**: `astroml-production`
- **Service Name**: `production`
- **Port**: `8004`
- **Profile**: `prod`
- **Purpose**: Production deployment
- **Features**:
  - Optimized image size
  - Production configuration
  - Health monitoring

### Training Services

#### GPU Training

- **Container**: `astroml-training-gpu`
- **Service Name**: `training-gpu`
- **TensorBoard Port**: `6006`
- **Profile**: `gpu`
- **GPU Required**: Yes
- **Purpose**: GPU-accelerated machine learning training
- **Volumes**:
  - `training_models`
  - `training_data`
  - `training_logs`

#### CPU Training

- **Container**: `astroml-training-cpu`
- **Service Name**: `training-cpu`
- **TensorBoard Port**: `6007`
- **Profile**: `cpu`
- **GPU Required**: No
- **Purpose**: CPU-based machine learning training
- **Volumes**:
  - `training_models`
  - `training_data`
  - `training_logs`

### Soroban Services

#### Soroban Development

- **Service Name**: `soroban-dev`
- **Profile**: `soroban`
- **Purpose**: Smart contract development environment
- **Features**:
  - Live contract development
  - Cargo watch support
  - Rapid iteration workflow

#### Soroban Build

- **Service Name**: `soroban-build`
- **Profile**: `soroban-build`
- **Purpose**: Build and package Soroban contracts for deployment

#### Soroban Testing

- **Service Name**: `soroban-test`
- **Profile**: `soroban-test`
- **Purpose**: Execute Soroban contract tests and validation suites

### Monitoring Services

#### Prometheus
### Monitoring Services

#### Prometheus
- **Container**: `astroml-prometheus`
- **Port**: `9090`
- **Profile**: `monitoring`
- **Purpose**: Metrics collection and monitoring

#### Grafana
- **Container**: `astroml-grafana`
- **Port**: `3000`
- **Profile**: `monitoring`
- **Purpose**: Dashboards and metrics visualization
- **Default Credentials**: `admin/admin`

---

## Docker Stages

### Main Dockerfile Stages

#### Base Stage
- Common Python runtime and dependencies
- Python 3.11
- Non-root `astroml` user
- Shared libraries and tooling

#### Ingestion Stage
- Data ingestion and streaming workloads
- Health checks enabled
- Default command:
```bash
python -m astroml.ingestion
```bash
# List volumes
docker volume ls

### Volume Management

```bash
# List volumes
docker volume ls

# Remove unused volumes
docker volume prune

# Backup PostgreSQL volume
docker run --rm \
  -v astroml_postgres_data:/data \
  -v $(pwd):/backup \
  ubuntu \
  tar czf /backup/postgres_backup.tar.gz /data

# Restore PostgreSQL volume
docker run --rm \
  -v astroml_postgres_data:/data \
  -v $(pwd):/backup \
  ubuntu \
  tar xzf /backup/postgres_backup.tar.gz -C /

# Recreate all project volumes
docker-compose down -v
docker-compose up -d
```

### Container Orchestration

```bash
# Scale services
docker-compose up -d --scale ingestion=3

# Update a service without downtime
docker-compose up -d --no-deps --build <service>

# Rolling update
docker-compose up -d --build --no-deps ingestion
```

### Debug Commands

#### Check Container Status

```bash
# Show running containers
docker-compose ps

# Inspect a specific container
docker inspect astroml-feature-store
```

#### Access Container Logs

```bash
# Show recent logs
docker-compose logs --tail=100 feature-store

# Follow logs in real time
docker-compose logs -f feature-store

# Show logs from the last hour
docker-compose logs --since="1h" feature-store
```

#### Health Checks

```bash
# Check service health
docker-compose ps

# Run a manual health check
docker-compose exec feature-store python -c "import astroml.features"
```

### Production Deployment

#### Build Production Image

```bash
docker-compose build production

docker tag astroml_production:latest your-registry/astroml:latest

docker push your-registry/astroml:latest
```

#### Deploy to Production

```bash
# Set production environment variables
export DATABASE_URL=production_db_url
export REDIS_URL=production_redis_url

# Start production services
docker-compose --profile prod up -d
```

### CI/CD Integration

#### GitHub Actions Example

```yaml
name: Docker Build and Test

on:
  - push
  - pull_request

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Build Docker images
        run: docker-compose build

      - name: Run tests
        run: docker-compose run --rm dev pytest

      - name: Build Soroban contracts
        run: docker-compose --profile soroban-build run soroban-build
```

### Security Best Practices

#### Scan Images for Vulnerabilities

```bash
# Scan with Trivy
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image astroml:latest

# Scan with Docker Scout
docker scout quickview astroml:latest
```

#### Use Non-Root Users

```dockerfile
RUN groupadd -r astroml && useradd -r -g astroml astroml

USER astroml
```

#### Limit Container Capabilities

```yaml
security_opt:
  - no-new-privileges:true

cap_drop:
  - ALL

cap_add:
  - NET_BIND_SERVICE
```
```

### Performance Optimization

## Performance Optimization

### Build Optimization

#### Use BuildKit

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with BuildKit
docker-compose build

# Use cache for faster builds
docker-compose build --no-cache=false

# Order instructions to maximize cache efficiency
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

# Builder stage
FROM python:3.11-slim as builder

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local

# Builder stage
FROM python:3.11-slim as builder

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Full system cleanup
docker system prune -a
```

### Backups

#### Database Backup

```bash
# Automated backup script
docker-compose exec postgres pg_dump -U astroml astroml > backup_$(date +%Y%m%d).sql
```

for vol in $(docker volume ls -q); do
  docker run --rm -v $vol:/data -v $(pwd):/backup \
    ubuntu tar czf /backup/${vol}.tar.gz /data
done

## Support

For issues or questions:
- GitHub Issues: https://github.com/jaynomyaro/astroml/issues
- Documentation: https://github.com/jaynomyaro/astroml/docs
- Docker Documentation: https://docs.docker.com

docker run --rm \
  -v astroml_postgres_data:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/postgres_backup.tar.gz -C /

deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

## Advanced Usage

### Custom Dockerfiles

Create custom Dockerfiles for specific use cases:

```dockerfile
# Custom Dockerfile for research
FROM astroml:development

# Install additional packages
RUN pip install jupyterlab-widgets plotly seaborn

# Copy research notebooks
COPY research/ /app/research/
```

docker run --rm \
  -v astroml_postgres_data:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/postgres_backup.tar.gz -C /

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
```

### Service Mesh

Integrate with service mesh (Istio, Linkerd):

`apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    sidecar.istio.io/inject: "true"``yaml
# Add service mesh annotations
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    sidecar.istio.io/inject: "true"
```
Security Considerations
Use non-root users
Limit container capabilities
Scan images for vulnerabilities
Use image signing
Use private networks
Enable TLS encryption
Configure firewall rules
Use secrets management
Perform regular audits
Support

If you face issues:

Check logs: docker-compose logs
Inspect containers: docker-compose ps
Search GitHub issues
Open a new issue with full details
## Security Considerations

### Container Security
- Use non-root users
- Limit container capabilities
- Scan images for vulnerabilities
- Use image signing

### Network Security
- Use private networks
- Implement TLS encryption
- Configure firewall rules
- Monitor network traffic

### Data Security
- Encrypt sensitive data
- Use secrets management
- Implement access controls
- Regular security audits

## Best Practices

### Development
- Use volume mounts for code changes
- Enable hot reloading
- Use development tools
- Write tests for all features

### Production
- Use specific image tags
- Implement health checks
- Use resource limits
- Monitor performance

# Backup all volumes
for vol in $(docker volume ls -q); do
  docker run --rm -v $vol:/data -v $(pwd):/backup ubuntu tar czf /backup/${vol}.tar.gz /data
done
Support
For issues or questions:

Check the local documentation and logs (docker-compose logs).

Review logs and error messages.

Search existing GitHub Issues.

Create a new issue with detailed replication steps.


Additional Resources
Docker Documentation

Docker Compose Documentation

AstroML Repository & Docs

Feature Store Documentation
