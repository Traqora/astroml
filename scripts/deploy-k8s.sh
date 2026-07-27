#!/bin/bash
# Kubernetes deployment script for AstroML
# This script handles deployment to Kubernetes clusters

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

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check kubectl
    if ! command -v kubectl > /dev/null 2>&1; then
        print_error "kubectl is not installed"
        exit 1
    fi
    print_status "kubectl is installed"
    
    # Check kustomize
    if ! command -v kustomize > /dev/null 2>&1; then
        print_warning "kustomize is not installed, installing..."
        curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
        sudo mv kustomize /usr/local/bin/
    fi
    print_status "kustomize is available"
    
    # Check cluster connectivity
    if ! kubectl cluster-info > /dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_status "Kubernetes cluster is accessible"
}

# Function to deploy to namespace
deploy_namespace() {
    local namespace=${1:-astroml}
    print_header "Deploying Namespace"
    
    kubectl create namespace $namespace --dry-run=client -o yaml | kubectl apply -f -
    print_status "Namespace $namespace created/verified"
}

# Function to deploy secrets
deploy_secrets() {
    print_header "Deploying Secrets"
    
    # Check if secrets file exists
    if [ -f "k8s/secrets.yaml" ]; then
        kubectl apply -f k8s/secrets.yaml
        print_status "Secrets deployed"
    else
        print_warning "No secrets file found, using default values"
    fi
}

# Function to deploy base infrastructure
deploy_base() {
    print_header "Deploying Base Infrastructure"
    
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/postgres-deployment.yaml
    kubectl apply -f k8s/redis-deployment.yaml
    
    print_status "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n astroml --timeout=300s
    
    print_status "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n astroml --timeout=300s
    
    print_status "Base infrastructure deployed"
}

# Function to deploy Feature Store
deploy_feature_store() {
    print_header "Deploying Feature Store"
    
    kubectl apply -f k8s/feature-store-deployment.yaml
    
    print_status "Waiting for Feature Store to be ready..."
    kubectl wait --for=condition=ready pod -l app=feature-store -n astroml --timeout=300s
    
    print_status "Feature Store deployed"
}

# Function to deploy applications
deploy_applications() {
    print_header "Deploying Applications"
    
    kubectl apply -f k8s/astroml-deployment.yaml
    kubectl apply -f k8s/services.yaml
    
    print_status "Waiting for applications to be ready..."
    kubectl wait --for=condition=ready pod -l app=astroml-ingestion -n astroml --timeout=300s
    kubectl wait --for=condition=ready pod -l app=astroml-training -n astroml --timeout=300s
    
    print_status "Applications deployed"
}

# Function to deploy monitoring
deploy_monitoring() {
    print_header "Deploying Monitoring Stack"
    
    kubectl apply -f k8s/monitoring.yaml
    
    print_status "Waiting for monitoring stack to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n astroml --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n astroml --timeout=300s
    
    print_status "Monitoring stack deployed"
}

# Function to deploy logging
deploy_logging() {
    print_header "Deploying Logging Stack"
    
    kubectl apply -f k8s/logging.yaml
    
    print_status "Waiting for logging stack to be ready..."
    kubectl wait --for=condition=ready pod -l app=elasticsearch -n astroml --timeout=300s
    kubectl wait --for=condition=ready pod -l app=kibana -n astroml --timeout=300s
    
    print_status "Logging stack deployed"
}

# Function to deploy ingress
deploy_ingress() {
    print_header "Deploying Ingress"
    
    kubectl apply -f k8s/ingress.yaml
    
    print_status "Ingress deployed"
}

# Function to deploy using kustomize
deploy_kustomize() {
    print_header "Deploying with Kustomize"
    
    kustomize build k8s/ | kubectl apply -f -
    
    print_status "Deployment completed with Kustomize"
}

# Function to verify deployment
verify_deployment() {
    print_header "Verifying Deployment"
    
    print_status "Checking pod status..."
    kubectl get pods -n astroml
    
    print_status "Checking services..."
    kubectl get services -n astroml
    
    print_status "Checking ingress..."
    kubectl get ingress -n astroml
    
    print_status "Deployment verification completed"
}

# Function to get access information
get_access_info() {
    print_header "Access Information"
    
    print_status "Service Endpoints:"
    kubectl get services -n astroml
    
    print_status "Ingress Endpoints:"
    kubectl get ingress -n astroml
    
    print_status "To access Grafana:"
    echo "kubectl port-forward -n astroml svc/grafana 3000:3000"
    
    print_status "To access Kibana:"
    echo "kubectl port-forward -n astroml svc/kibana 5601:5601"
}

# Function to rollback deployment
rollback_deployment() {
    local deployment=${1:-astroml-ingestion}
    print_header "Rolling Back Deployment"
    
    kubectl rollout undo deployment/$deployment -n astroml
    
    print_status "Rollback completed for $deployment"
}

# Function to scale deployment
scale_deployment() {
    local deployment=${1:-astroml-ingestion}
    local replicas=${2:-3}
    print_header "Scaling Deployment"
    
    kubectl scale deployment/$deployment -n astroml --replicas=$replicas
    
    print_status "Deployment $deployment scaled to $replicas replicas"
}

# Function to show logs
show_logs() {
    local deployment=${1:-astroml-ingestion}
    print_header "Showing Logs"
    
    kubectl logs -f deployment/$deployment -n astroml
}

# Function to clean up
cleanup() {
    print_header "Cleaning Up"
    
    kustomize build k8s/ | kubectl delete -f -
    
    print_status "Cleanup completed"
}

# Main execution
main() {
    local command=${1:-deploy}
    local environment=${2:-production}
    
    print_header "AstroML Kubernetes Deployment"
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    # Check prerequisites
    check_prerequisites
    
    case $command in
        "deploy")
            deploy_namespace
            deploy_secrets
            deploy_base
            deploy_feature_store
            deploy_applications
            deploy_monitoring
            deploy_logging
            deploy_ingress
            verify_deployment
            get_access_info
            ;;
        "kustomize")
            deploy_kustomize
            verify_deployment
            get_access_info
            ;;
        "monitoring")
            deploy_monitoring
            ;;
        "logging")
            deploy_logging
            ;;
        "verify")
            verify_deployment
            ;;
        "access")
            get_access_info
            ;;
        "rollback")
            rollback_deployment $2
            ;;
        "scale")
            scale_deployment $2 $3
            ;;
        "logs")
            show_logs $2
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            echo "AstroML Kubernetes Deployment Script"
            echo ""
            echo "Usage: $0 [COMMAND] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  deploy          Deploy all components"
            echo "  kustomize       Deploy using Kustomize"
            echo "  monitoring      Deploy monitoring stack only"
            echo "  logging         Deploy logging stack only"
            echo "  verify          Verify deployment status"
            echo "  access          Show access information"
            echo "  rollback [name] Rollback deployment"
            echo "  scale [name] [replicas] Scale deployment"
            echo "  logs [name]     Show logs for deployment"
            echo "  cleanup         Remove all components"
            echo "  help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 deploy"
            echo "  $0 kustomize"
            echo "  $0 scale astroml-ingestion 5"
            echo "  $0 logs feature-store"
            ;;
    esac
}

# Handle signals gracefully
trap 'print_warning "Deployment interrupted"; exit 1' SIGINT SIGTERM

# Execute main function
main "$@"
