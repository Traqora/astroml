#!/bin/bash
# Docker development script for AstroML
# This script provides convenient commands for Docker development

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
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1 && ! docker compose version > /dev/null 2>&1; then
        print_error "docker-compose is not installed or not in PATH."
        exit 1
    fi
}

# Function to get docker-compose command
get_docker_compose_cmd() {
    if command -v docker-compose > /dev/null 2>&1; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

# Function to build Docker images
build_images() {
    print_header "Building Docker Images"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    print_status "Building base image..."
    $COMPOSE_CMD build base
    
    print_status "Building development image..."
    $COMPOSE_CMD build development
    
    print_status "Building Feature Store image..."
    $COMPOSE_CMD build feature-store
    
    print_status "Building ingestion image..."
    $COMPOSE_CMD build ingestion
    
    print_status "Building training images..."
    $COMPOSE_CMD build training-cpu
    
    print_status "All images built successfully!"
}

# Function to start development environment
start_dev() {
    print_header "Starting Development Environment"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    # Start core services
    print_status "Starting PostgreSQL and Redis..."
    $COMPOSE_CMD up -d postgres redis
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Start development environment
    print_status "Starting development container..."
    $COMPOSE_CMD --profile dev up -d
    
    print_status "Development environment started!"
    print_status "Jupyter Lab: http://localhost:8888"
    print_status "TensorBoard: http://localhost:6008"
    
    # Show logs
    $COMPOSE_CMD logs -f dev
}

# Function to start Feature Store
start_feature_store() {
    print_header "Starting Feature Store"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    # Start core services
    print_status "Starting PostgreSQL and Redis..."
    $COMPOSE_CMD up -d postgres redis
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Start Feature Store
    print_status "Starting Feature Store..."
    $COMPOSE_CMD --profile feature-store up -d
    
    print_status "Feature Store started!"
    print_status "Feature Store API: http://localhost:8000"
    
    # Show logs
    $COMPOSE_CMD logs -f feature-store
}

# Function to start full environment
start_full() {
    print_header "Starting Full Environment"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    # Start all services
    print_status "Starting all services..."
    $COMPOSE_CMD --profile full up -d
    
    print_status "Full environment started!"
    print_status "Feature Store: http://localhost:8000"
    print_status "Ingestion: http://localhost:8001"
    print_status "Streaming: http://localhost:8002"
    print_status "Development: http://localhost:8003"
    print_status "Production: http://localhost:8004"
    print_status "Jupyter Lab: http://localhost:8888"
    print_status "TensorBoard: http://localhost:6008"
    
    # Show logs
    $COMPOSE_CMD logs -f
}

# Function to run tests
run_tests() {
    print_header "Running Tests"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    # Start services needed for tests
    print_status "Starting test dependencies..."
    $COMPOSE_CMD up -d postgres redis
    
    # Wait for services to be ready
    sleep 10
    
    # Run tests
    print_status "Running test suite..."
    $COMPOSE_CMD run --rm development pytest tests/ -v --cov=astroml --cov-report=html
    
    print_status "Tests completed!"
    print_status "Coverage report: htmlcov/index.html"
}

# Function to run Feature Store tests
run_feature_store_tests() {
    print_header "Running Feature Store Tests"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    # Start services needed for tests
    print_status "Starting test dependencies..."
    $COMPOSE_CMD up -d postgres redis
    
    # Wait for services to be ready
    sleep 10
    
    # Run Feature Store tests
    print_status "Running Feature Store test suite..."
    $COMPOSE_CMD run --rm development pytest tests/features/ -v --cov=astroml.features --cov-report=html
    
    print_status "Feature Store tests completed!"
    print_status "Coverage report: htmlcov/index.html"
}

# Function to stop services
stop_services() {
    print_header "Stopping Services"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    print_status "Stopping all services..."
    $COMPOSE_CMD down
    
    print_status "All services stopped!"
}

# Function to clean up
cleanup() {
    print_header "Cleaning Up"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    print_status "Stopping and removing containers..."
    $COMPOSE_CMD down -v --remove-orphans
    
    print_status "Removing images..."
    $COMPOSE_CMD down --rmi all
    
    print_status "Removing volumes..."
    docker volume prune -f
    
    print_status "Cleanup completed!"
}

# Function to show logs
show_logs() {
    local service=${1:-}
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        $COMPOSE_CMD logs -f
    else
        print_status "Showing logs for $service..."
        $COMPOSE_CMD logs -f "$service"
    fi
}

# Function to execute commands in container
exec_container() {
    local service=${1:-development}
    shift
    local command="$@"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    if [ -z "$command" ]; then
        print_status "Opening shell in $service container..."
        $COMPOSE_CMD exec "$service" /bin/bash
    else
        print_status "Executing command in $service container..."
        $COMPOSE_CMD exec "$service" $command
    fi
}

# Function to show status
show_status() {
    print_header "Service Status"
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    
    $COMPOSE_CMD ps
    
    echo ""
    print_header "Port Mappings"
    echo "Feature Store: http://localhost:8000"
    echo "Ingestion: http://localhost:8001"
    echo "Streaming: http://localhost:8002"
    echo "Development: http://localhost:8003"
    echo "Production: http://localhost:8004"
    echo "PostgreSQL: localhost:5432"
    echo "Redis: localhost:6379"
    echo "Jupyter Lab: http://localhost:8888"
    echo "TensorBoard: http://localhost:6008"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000"
}

# Function to show help
show_help() {
    echo "AstroML Docker Development Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker images"
    echo "  dev                Start development environment"
    echo "  feature-store      Start Feature Store only"
    echo "  full               Start full environment"
    echo "  test               Run test suite"
    echo "  test-feature-store Run Feature Store tests"
    echo "  stop               Stop all services"
    echo "  cleanup            Clean up containers, images, and volumes"
    echo "  logs [service]     Show logs (all services or specific service)"
    echo "  exec [service] [cmd] Execute command in container"
    echo "  status             Show service status"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev                    # Start development environment"
    echo "  $0 exec dev bash          # Open shell in development container"
    echo "  $0 exec dev pytest tests/ # Run tests in development container"
    echo "  $0 logs feature-store     # Show Feature Store logs"
    echo "  $0 test                   # Run all tests"
}

# Main execution
main() {
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    # Parse command
    case "${1:-help}" in
        "build")
            build_images
            ;;
        "dev")
            start_dev
            ;;
        "feature-store")
            start_feature_store
            ;;
        "full")
            start_full
            ;;
        "test")
            run_tests
            ;;
        "test-feature-store")
            run_feature_store_tests
            ;;
        "stop")
            stop_services
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            show_logs "$2"
            ;;
        "exec")
            exec_container "$2" "${@:3}"
            ;;
        "status")
            show_status
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Execute main function
main "$@"
