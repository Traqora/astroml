# Docker entrypoint script for AstroML Ingestion Service
# This script initializes the database and starts the ingestion service

#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[INFO]${NC} Starting AstroML Ingestion Service"

# Function to wait for database
wait_for_db() {
    echo -e "${YELLOW}[WAIT]${NC} Waiting for PostgreSQL to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" > /dev/null 2>&1; then
            echo -e "${GREEN}[INFO]${NC} PostgreSQL is ready"
            return 0
        fi
        
        echo -e "${YELLOW}[WAIT]${NC} PostgreSQL not ready yet. Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}[ERROR]${NC} PostgreSQL failed to become ready"
    return 1
}

# Function to wait for Redis
wait_for_redis() {
    echo -e "${YELLOW}[WAIT]${NC} Waiting for Redis to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
            echo -e "${GREEN}[INFO]${NC} Redis is ready"
            return 0
        fi
        
        echo -e "${YELLOW}[WAIT]${NC} Redis not ready yet. Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}[ERROR]${NC} Redis failed to become ready"
    return 1
}

# Wait for dependent services
wait_for_db
wait_for_redis

# Run database migrations
echo -e "${GREEN}[INFO]${NC} Running database migrations..."
if command -v alembic &> /dev/null; then
    cd /app && alembic upgrade head || echo -e "${YELLOW}[WARN]${NC} Migrations may have already been applied"
else
    echo -e "${YELLOW}[WARN]${NC} Alembic not found, skipping migrations"
fi

# Start the ingestion service
echo -e "${GREEN}[INFO]${NC} Starting ingestion service..."

# Graceful shutdown — allow in-flight ingestion to finish (issue #775)
trap 'echo -e "${YELLOW}[INFO]${NC} Received shutdown signal, stopping..."; exit 0' SIGTERM SIGINT

exec python -m astroml.ingestion
