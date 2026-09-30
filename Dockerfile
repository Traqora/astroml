# Multi-stage Dockerfile for AstroML with Feature Store
# This Dockerfile creates optimized images for development, testing, and production
# Includes comprehensive Feature Store implementation with caching and versioning

# ============================================================================
# BASE STAGE - Common dependencies and Python environment
# ============================================================================
# Pin the Python base image to an exact patch + distro (#196) so a rebuild
# six months from now produces the same intermediate layers. The slim
# bookworm tag is roughly 60% smaller than the default `python:3.11` image.
FROM python:3.11.9-slim-bookworm AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ASTROML_ENV=container \
    FEATURE_STORE_PATH=/app/feature_store

# Install system dependencies. `--no-install-recommends` skips the long tail
# of suggested packages (man-db, locales, etc.) that ship with apt's default
# recommend resolution and add ~80MB to the image (#196).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    postgresql-client \
    redis-tools \
    netcat-openbsd \
    jq \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r astroml && useradd -r -g astroml astroml

# Set working directory
WORKDIR /app

# Copy and install Python dependencies.
# This installs the full stack (requirements.txt) intentionally — this base
# stage backs both the ingestion and training images below. For a
# non-container install that only needs one purpose (dev tooling / training
# only / the API service), see the smaller requirements-*.txt files and the
# decision tree in REQUIREMENTS.md (issue #561).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================================================
# INGESTION STAGE - Optimized for data ingestion and streaming with Feature Store
# ============================================================================
FROM base AS ingestion

# Install additional dependencies for ingestion.
RUN apt-get update && apt-get install -y --no-install-recommends \
    jq \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml migrations/ ./migrations/
COPY --chown=astroml:astroml docs/ ./docs/
COPY --chown=astroml:astroml examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/feature_store && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose ports for health checks and monitoring
EXPOSE 8000 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import astroml.ingestion; import astroml.features" || exit 1

# Default command for ingestion
CMD ["python", "-m", "astroml.ingestion"]

# ============================================================================
# TRAINING STAGE - Optimized for ML training with GPU support
# ============================================================================
FROM nvidia/cuda:12.1-runtime-base-ubuntu22.04 AS training-base

# Install Python and system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    curl \
    git \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0

# Create app user
RUN groupadd -r astroml && useradd -r -g astroml astroml

# Set working directory
WORKDIR /app

# Copy and install Python dependencies. See the base stage above for why
# this stays on the full requirements.txt rather than requirements-train.txt
# — the CUDA-specific torch install below then overrides the CPU/generic
# torch this pulls in. If you're bumping the torch pin, update it in
# requirements.txt / -cpu.txt / -train.txt and here together — see
# "Upgrading a major dependency" in REQUIREMENTS.md (issue #562).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric with CUDA support
RUN pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml docs/ ./docs/
COPY --chown=astroml:astroml examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/feature_store && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose port for monitoring (TensorBoard, etc.)
EXPOSE 6006

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric; import astroml.features" || exit 1

# Default command for training
CMD ["python", "-m", "astroml.training.train_gcn"]

# ============================================================================
# CPU-ONLY TRAINING STAGE - For environments without GPU
# ============================================================================
FROM base as training-cpu

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml docs/ ./docs/
COPY --chown=astroml:astroml examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/feature_store && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose port for monitoring
EXPOSE 6006

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric; import astroml.features" || exit 1

# Default command for training
CMD ["python", "-m", "astroml.training.train_gcn"]

# ============================================================================
# DEVELOPMENT STAGE - Includes development tools and testing
# ============================================================================
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black flake8 mypy jupyter

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml tests/ ./tests/
COPY --chown=astroml:astroml migrations/ ./migrations/
COPY --chown=astroml:astroml docs/ ./docs/
COPY --chown=astroml:astroml examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/notebooks /app/feature_store && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose ports for development
EXPOSE 8000 8080 8888 6006

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ============================================================================
# FEATURE STORE STAGE - Dedicated Feature Store service
# ============================================================================
FROM base as feature-store

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml docs/ ./docs/
COPY --chown=astroml:astroml examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/feature_store && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose ports for Feature Store API
EXPOSE 8000 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import astroml.features; from astroml.features import create_feature_store" || exit 1

# Default command for Feature Store service
CMD ["python", "-c", "from astroml.features import create_feature_store; store = create_feature_store('/app/feature_store'); print('Feature Store service ready')"]

# ============================================================================
# PRODUCTION STAGE - Minimal production image
# ============================================================================
FROM base as production

# Copy only necessary files for production
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml docs/ ./docs/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/feature_store && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import astroml; import astroml.features" || exit 1

# Default production command (can be overridden)
CMD ["python", "-m", "astroml.ingestion"]

# ============================================================================
# TRAINING STAGE - Alias for training with GPU (uses training-base)
# ============================================================================
FROM training-base as training

# This stage is used when GPU is available
CMD ["python", "-m", "astroml.ingestion"]
