#!/bin/bash
# Docker verification script for AstroML
# This script tests the Docker setup and verifies all services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if Docker is running
check_docker() {
    print_header "Checking Docker"
    
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running"
        return 1
    fi
    
    print_status "Docker is running"
    docker --version
    return 0
}

# Function to check docker-compose
check_docker_compose() {
    print_header "Checking Docker Compose"
    
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
        print_status "Using docker-compose"
        docker-compose --version
    elif docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
        print_status "Using docker compose"
        docker compose version
    else
        print_error "docker-compose is not available"
        return 1
    fi
    
    return 0
}

# Function to verify Docker images
verify_images() {
    print_header "Verifying Docker Images"
    
    local images=(
        "astroml_base"
        "astroml_development"
        "astroml_feature-store"
        "astroml_ingestion"
        "astroml_training-cpu"
        "astroml_production"
    )
    
    for image in "${images[@]}"; do
        if docker images | grep -q "$image"; then
            print_status "✓ $image image exists"
        else
            print_warning "✗ $image image not found"
        fi
    done
}

# Function to verify Docker volumes
verify_volumes() {
    print_header "Verifying Docker Volumes"
    
    local volumes=(
        "astroml_postgres_data"
        "astroml_redis_data"
        "astroml_feature_store_data"
        "astroml_feature_store_logs"
    )
    
    for volume in "${volumes[@]}"; do
        if docker volume ls | grep -q "$volume"; then
            print_status "✓ $volume volume exists"
        else
            print_warning "✗ $volume volume not found"
        fi
    done
}

# Function to test core services
test_core_services() {
    print_header "Testing Core Services"
    
    # Start core services
    print_status "Starting core services..."
    $COMPOSE_CMD up -d postgres redis
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    sleep 15
    
    # Test PostgreSQL
    print_status "Testing PostgreSQL connection..."
    if $COMPOSE_CMD exec -T postgres pg_isready -U astroml -d astroml; then
        print_status "✓ PostgreSQL is ready"
    else
        print_error "✗ PostgreSQL connection failed"
    fi
    
    # Test Redis
    print_status "Testing Redis connection..."
    if $COMPOSE_CMD exec -T redis redis-cli ping | grep -q "PONG"; then
        print_status "✓ Redis is ready"
    else
        print_error "✗ Redis connection failed"
    fi
}

# Function to test API service
test_api_service() {
    print_header "Testing API Service"
    
    # Start API service
    print_status "Starting API service..."
    $COMPOSE_CMD up -d api
    
    # Wait for API to start
    print_status "Waiting for API service to start..."
    sleep 25
    
    # Test API health endpoint
    print_status "Testing API health endpoint..."
    if curl -s http://localhost:8000/health | grep -q "ok"; then
        print_status "✓ API health endpoint is responding"
    else
        print_error "✗ API health endpoint failed"
    fi
    
    # Test API transactions endpoint
    print_status "Testing API transactions endpoint..."
    if curl -s http://localhost:8000/api/v1/transactions/stats | grep -q "total_count"; then
        print_status "✓ API transactions endpoint is responding"
    else
        print_warning "✗ API transactions endpoint not responding (may need database data)"
    fi
    
    # Test API accounts endpoint
    print_status "Testing API accounts endpoint..."
    if curl -s http://localhost:8000/api/v1/accounts | grep -q "total"; then
        print_status "✓ API accounts endpoint is responding"
    else
        print_warning "✗ API accounts endpoint not responding (may need database data)"
    fi
}

# Function to test Feature Store
test_feature_store() {
    print_header "Testing Feature Store"
    
    # Start Feature Store
    print_status "Starting Feature Store..."
    $COMPOSE_CMD up -d feature-store
    
    # Wait for Feature Store to start
    print_status "Waiting for Feature Store to start..."
    sleep 20
    
    # Test Feature Store import
    print_status "Testing Feature Store import..."
    if $COMPOSE_CMD exec -T feature-store python -c "
import astroml.features
from astroml.features import create_feature_store
store = create_feature_store('/app/feature_store')
print('Feature Store initialized successfully')
"; then
        print_status "✓ Feature Store is working"
    else
        print_error "✗ Feature Store failed to initialize"
    fi
    
    # Test Feature Store functionality
    print_status "Testing Feature Store functionality..."
    if $COMPOSE_CMD exec -T feature-store python -c "
from astroml.features import create_feature_store, FeatureType
import pandas as pd
import numpy as np

# Create test feature
def test_computer(data, entity_col, timestamp_col, **kwargs):
    return pd.DataFrame({'test_feature': [1, 2, 3]})

store = create_feature_store('/app/feature_store')
feature_def = store.register_feature(
    name='test_feature',
    computer=test_computer,
    description='Test feature',
    feature_type=FeatureType.NUMERIC
)
print('Feature registration successful')
"; then
        print_status "✓ Feature Store functionality working"
    else
        print_error "✗ Feature Store functionality failed"
    fi
}

# Function to test development environment
test_development() {
    print_header "Testing Development Environment"
    
    # Start development environment
    print_status "Starting development environment..."
    $COMPOSE_CMD up -d dev
    
    # Wait for development environment to start
    print_status "Waiting for development environment to start..."
    sleep 20
    
    # Test Jupyter Lab
    print_status "Testing Jupyter Lab..."
    if curl -s http://localhost:8888 | grep -q "Jupyter"; then
        print_status "✓ Jupyter Lab is accessible"
    else
        print_warning "✗ Jupyter Lab not accessible (may need more time)"
    fi
    
    # Test Python environment
    print_status "Testing Python environment..."
    if $COMPOSE_CMD exec -T dev python -c "
import astroml
import astroml.features
import pandas as pd
import numpy as np
import torch
import networkx
print('All Python packages imported successfully')
"; then
        print_status "✓ Python environment is working"
    else
        print_error "✗ Python environment failed"
    fi
}

# Function to run tests
run_tests() {
    print_header "Running Tests"
    
    # Run Feature Store tests
    print_status "Running Feature Store tests..."
    if $COMPOSE_CMD exec -T dev pytest tests/features/ -v --tb=short; then
        print_status "✓ Feature Store tests passed"
    else
        print_error "✗ Feature Store tests failed"
    fi
    
    # Run basic tests
    print_status "Running basic tests..."
    if $COMPOSE_CMD exec -T dev pytest tests/validation/test_data_quality.py -v --tb=short; then
        print_status "✓ Basic tests passed"
    else
        print_error "✗ Basic tests failed"
    fi
}

# Function to test ports
test_ports() {
    print_header "Testing Port Accessibility"
    
    local ports=(
        "8000:API Service"
        "8001:Ingestion"
        "8002:Streaming"
        "8003:Development"
        "8888:Jupyter Lab"
        "6008:TensorBoard"
        "5432:PostgreSQL"
        "6379:Redis"
    )
    
    for port_info in "${ports[@]}"; do
        port=$(echo $port_info | cut -d: -f1)
        service=$(echo $port_info | cut -d: -f2)
        
        if nc -z localhost $port 2>/dev/null; then
            print_status "✓ $service (port $port) is accessible"
        else
            print_warning "✗ $service (port $port) not accessible"
        fi
    done
}

# Function to test logs
test_logs() {
    print_header "Testing Logs"
    
    local services=(
        "postgres"
        "redis"
        "api"
        "feature-store"
        "dev"
    )
    
    for service in "${services[@]}"; do
        if $COMPOSE_CMD logs $service | grep -q "ERROR\|CRITICAL"; then
            print_warning "⚠ $service has errors in logs"
        else
            print_status "✓ $service logs look clean"
        fi
    done
}

# Function to cleanup
cleanup() {
    print_header "Cleaning Up"
    
    print_status "Stopping all services..."
    $COMPOSE_CMD down
    
    print_status "Cleanup completed"
}

# Function to generate report
generate_report() {
    print_header "Verification Report"
    
    echo "Docker Setup Verification completed on $(date)"
    echo "=========================================="
    echo ""
    echo "Services Tested:"
    echo "- PostgreSQL Database"
    echo "- Redis Cache"
    echo "- API Service"
    echo "- Feature Store"
    echo "- Development Environment"
    echo "- Python Environment"
    echo "- Port Accessibility"
    echo "- Test Suite"
    echo ""
    echo "For detailed logs, check the output above."
    echo ""
    echo "Next Steps:"
    echo "1. Start development: ./scripts/docker-dev.sh dev"
    echo "2. Access Jupyter Lab: http://localhost:8888"
    echo "3. Access API: http://localhost:8000"
    echo "4. Access API docs: http://localhost:8000/docs"
    echo "5. Run Feature Store example: docker-compose exec dev python examples/feature_store_example.py"
    echo "6. Run tests: ./scripts/docker-dev.sh test"
}

# Main execution
main() {
    print_header "AstroML Docker Verification"
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    # Run verification steps
    local failed_steps=0
    
    check_docker || ((failed_steps++))
    check_docker_compose || ((failed_steps++))
    verify_images
    verify_volumes
    test_core_services || ((failed_steps++))
    test_api_service || ((failed_steps++))
    test_feature_store || ((failed_steps++))
    test_development || ((failed_steps++))
    run_tests || ((failed_steps++))
    test_ports
    test_logs
    
    # Cleanup
    cleanup
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    if [ $failed_steps -eq 0 ]; then
        print_status "✅ All verification steps passed!"
        exit 0
    else
        print_error "❌ $failed_steps verification steps failed"
        exit 1
    fi
}

# Handle signals gracefully
trap 'print_warning "Verification interrupted"; cleanup; exit 1' SIGINT SIGTERM

# Execute main function
main "$@"
