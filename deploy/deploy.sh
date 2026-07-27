#!/bin/bash
# AstroML Production Deployment Script
# Usage: ./deploy.sh [start|stop|restart|status|logs]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            warn ".env file not found. Copying from .env.example"
            cp .env.example .env
            warn "Please edit .env and set POSTGRES_PASSWORD before continuing"
            exit 1
        else
            error ".env file not found. Please create one with required variables"
        fi
    fi
    
    # Source .env
    source .env
    
    # Check required variables
    if [ -z "$POSTGRES_PASSWORD" ]; then
        error "POSTGRES_PASSWORD is not set in .env"
    fi
}

# Start services
start() {
    log "Starting AstroML production services..."
    check_env
    
    docker compose -f docker-compose.prod.yml up -d
    
    log "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    check_health
    
    log "AstroML production services started successfully!"
    log "Feature Store API: http://localhost:${FEATURE_STORE_PORT:-8000}"
    log "PostgreSQL: localhost:${POSTGRES_PORT:-5432}"
    log "Redis: localhost:${REDIS_PORT:-6379}"
}

# Stop services
stop() {
    log "Stopping AstroML production services..."
    docker compose -f docker-compose.prod.yml down
    log "Services stopped"
}

# Restart services
restart() {
    stop
    start
}

# Check service health
check_health() {
    log "Checking service health..."
    
    # PostgreSQL
    if docker compose -f docker-compose.prod.yml exec -T postgres pg_isready -U ${POSTGRES_USER:-astroml} > /dev/null 2>&1; then
        log "✅ PostgreSQL is healthy"
    else
        warn "⚠️  PostgreSQL is not ready"
    fi
    
    # Redis
    if docker compose -f docker-compose.prod.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        log "✅ Redis is healthy"
    else
        warn "⚠️  Redis is not ready"
    fi
    
    # Feature Store
    if curl -s http://localhost:${FEATURE_STORE_PORT:-8000}/health > /dev/null 2>&1; then
        log "✅ Feature Store is healthy"
    else
        warn "⚠️  Feature Store is not ready"
    fi
}

# Show status
status() {
    log "Service status:"
    docker compose -f docker-compose.prod.yml ps
}

# Show logs
logs() {
    docker compose -f docker-compose.prod.yml logs -f
}

# Main
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
