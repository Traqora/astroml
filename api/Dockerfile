# Multi-stage Dockerfile for AstroML FastAPI Service
# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file.
# root-level `requirements-api.txt` (issue #561) is a discoverability alias
# for this same file (`-r api/requirements.txt`) for non-container installs
# — this build context intentionally keeps using the file directly.
COPY api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group for runtime security (issue #1249)
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/false appuser

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup astroml/ ./astroml/
COPY --chown=appuser:appgroup migrations/ ./migrations/

# Create logs directory
RUN mkdir -p /app/logs && chown appuser:appgroup /app/logs

# Expose ports
EXPOSE 8000 8443

# Health check — readiness probe (issue #775, #569)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/healthz/ready || exit 1

# Switch to non-root user before running (issue #1249)
USER appuser

# Run the application
# For production with HTTPS, set environment variables:
# HTTPS_ENABLED=true
# HTTPS_ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
# HSTS_ENABLED=true
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
