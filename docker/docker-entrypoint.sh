#!/bin/bash
# Docker entrypoint script for AstroML
# This script handles initialization and startup of AstroML services

set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to wait for a service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local timeout=${4:-30}
    
    log "Waiting for $service to be ready..."
    
    for i in $(seq 1 $timeout); do
        if nc -z $host $port; then
            log "$service is ready!"
            return 0
        fi
        log "Waiting for $service... ($i/$timeout)"
        sleep 1
    done
    
    log "ERROR: $service not ready after $timeout seconds"
    exit 1
}

# Function to initialize database
init_database() {
    log "Initializing database..."
    
    # Wait for PostgreSQL
    wait_for_service postgres 5432 "PostgreSQL"
    
    # Run migrations if they exist
    if [ -d "/app/migrations" ]; then
        log "Running database migrations..."
        cd /app
        python -m alembic upgrade head
    fi
    
    log "Database initialization complete"
}

# Function to initialize Feature Store
init_feature_store() {
    log "Initializing Feature Store..."
    
    # Create Feature Store directory if it doesn't exist
    mkdir -p /app/feature_store
    
    # Initialize Feature Store database
    cd /app
    python -c "
from astroml.features import create_feature_store
store = create_feature_store('/app/feature_store')
print('Feature Store initialized successfully')
"
    
    log "Feature Store initialization complete"
}

# Function to setup logging
setup_logging() {
    log "Setting up logging..."
    
    # Create log directories
    mkdir -p /app/logs
    
    # Set log level
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    
    log "Logging setup complete"
}

# Function to run health checks
health_check() {
    log "Running health checks..."
    
    # Check Python imports
    python -c "
import astroml
import astroml.features
print('Core modules imported successfully')
"
    
    # Check database connection
    python -c "
import sqlalchemy
engine = sqlalchemy.create_engine('$DATABASE_URL')
with engine.connect() as conn:
    conn.execute(sqlalchemy.text('SELECT 1'))
print('Database connection successful')
"
    
    # Check Redis connection if configured
    if [ -n "$REDIS_URL" ]; then
        python -c "
import redis
r = redis.from_url('$REDIS_URL')
r.ping()
print('Redis connection successful')
"
    fi
    
    log "Health checks passed"
}

# Function to start service
start_service() {
    local service_type=${1:-ingestion}
    
    log "Starting $service_type service..."
    
    case $service_type in
        "ingestion")
            exec python -m astroml.ingestion
            ;;
        "streaming")
            exec python -m astroml.ingestion.enhanced_stream
            ;;
        "training")
            exec python -m astroml.training.train_gcn
            ;;
        "feature-store")
            exec python -c "
from astroml.features import create_feature_store
store = create_feature_store('/app/feature_store')
print('Feature Store service ready')
import time
while True:
    time.sleep(60)
"
            ;;
        "development")
            # Start Jupyter Lab
            exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
            ;;
        "production")
            exec python -m astroml.ingestion
            ;;
        *)
            log "Unknown service type: $service_type"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    log "Starting AstroML Docker entrypoint..."
    
    # Setup logging
    setup_logging
    
    # Initialize database
    init_database
    
    # Initialize Feature Store
    init_feature_store
    
    # Run health checks
    health_check
    
    # Start the requested service
    start_service "$1"
}

# Handle signals gracefully
trap 'log "Received shutdown signal, exiting..."; exit 0' SIGTERM SIGINT

# Execute main function
main "$@"
