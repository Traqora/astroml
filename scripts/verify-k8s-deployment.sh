#!/bin/bash
# Kubernetes deployment verification script for AstroML
# This script verifies that all Kubernetes components are deployed correctly

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
        return 1
    fi
    print_status "kubectl is installed"
    
    # Check cluster connectivity
    if ! kubectl cluster-info > /dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    print_status "Kubernetes cluster is accessible"
    
    # Check kustomize
    if ! command -v kustomize > /dev/null 2>&1; then
        print_warning "kustomize is not installed"
    else
        print_status "kustomize is available"
    fi
    
    return 0
}

# Function to verify namespace
verify_namespace() {
    print_header "Verifying Namespace"
    
    if kubectl get namespace astroml > /dev/null 2>&1; then
        print_status "Namespace astroml exists"
        kubectl get namespace astroml
    else
        print_error "Namespace astroml does not exist"
        return 1
    fi
}

# Function to verify deployments
verify_deployments() {
    print_header "Verifying Deployments"
    
    local deployments=(
        "postgres"
        "redis"
        "feature-store"
        "astroml-ingestion"
        "astroml-training"
        "prometheus"
        "grafana"
        "elasticsearch"
        "kibana"
    )
    
    local failed_deployments=0
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment $deployment -n astroml > /dev/null 2>&1; then
            local ready=$(kubectl get deployment $deployment -n astroml -o jsonpath='{.status.readyReplicas}')
            local desired=$(kubectl get deployment $deployment -n astroml -o jsonpath='{.spec.replicas}')
            
            if [ "$ready" = "$desired" ] && [ "$ready" != "" ]; then
                print_status "✓ $deployment is ready ($ready/$desired replicas)"
            else
                print_warning "⚠ $deployment is not ready ($ready/$desired replicas)"
                failed_deployments=$((failed_deployments + 1))
            fi
        else
            print_warning "✗ $deployment does not exist"
            failed_deployments=$((failed_deployments + 1))
        fi
    done
    
    return $failed_deployments
}

# Function to verify pods
verify_pods() {
    print_header "Verifying Pods"
    
    print_status "Pod status in astroml namespace:"
    kubectl get pods -n astroml
    
    local failed_pods=0
    
    # Check for failed pods
    local failed=$(kubectl get pods -n astroml -o json | jq -r '.items[] | select(.status.phase=="Failed") | .metadata.name')
    if [ -n "$failed" ]; then
        print_error "Failed pods detected: $failed"
        failed_pods=$((failed_pods + 1))
    fi
    
    # Check for pending pods
    local pending=$(kubectl get pods -n astroml -o json | jq -r '.items[] | select(.status.phase=="Pending") | .metadata.name')
    if [ -n "$pending" ]; then
        print_warning "Pending pods detected: $pending"
    fi
    
    return $failed_pods
}

# Function to verify services
verify_services() {
    print_header "Verifying Services"
    
    print_status "Services in astroml namespace:"
    kubectl get services -n astroml
    
    local services=(
        "postgres"
        "redis"
        "feature-store"
        "astroml-ingestion"
        "astroml-training"
        "prometheus"
        "grafana"
        "elasticsearch"
        "kibana"
    )
    
    local failed_services=0
    
    for service in "${services[@]}"; do
        if kubectl get service $service -n astroml > /dev/null 2>&1; then
            local type=$(kubectl get service $service -n astroml -o jsonpath='{.spec.type}')
            local ports=$(kubectl get service $service -n astroml -o jsonpath='{.spec.ports[*].port}')
            print_status "✓ $service exists ($type, ports: $ports)"
        else
            print_warning "✗ $service does not exist"
            failed_services=$((failed_services + 1))
        fi
    done
    
    return $failed_services
}

# Function to verify ingress
verify_ingress() {
    print_header "Verifying Ingress"
    
    if kubectl get ingress -n astroml > /dev/null 2>&1; then
        print_status "Ingress resources in astroml namespace:"
        kubectl get ingress -n astroml
        return 0
    else
        print_warning "No ingress resources found"
        return 1
    fi
}

# Function to verify persistent volumes
verify_persistent_volumes() {
    print_header "Verifying Persistent Volumes"
    
    print_status "PVCs in astroml namespace:"
    kubectl get pvc -n astroml
    
    local pvcs=(
        "postgres-storage"
        "feature-store-pvc"
        "prometheus-pvc"
        "grafana-pvc"
        "elasticsearch-pvc"
    )
    
    local failed_pvcs=0
    
    for pvc in "${pvcs[@]}"; do
        if kubectl get pvc $pvc -n astroml > /dev/null 2>&1; then
            local status=$(kubectl get pvc $pvc -n astroml -o jsonpath='{.status.phase}')
            print_status "✓ $pvc exists ($status)"
        else
            print_warning "✗ $pvc does not exist"
            failed_pvcs=$((failed_pvcs + 1))
        fi
    done
    
    return $failed_pvcs
}

# Function to verify configmaps
verify_configmaps() {
    print_header "Verifying ConfigMaps"
    
    print_status "ConfigMaps in astroml namespace:"
    kubectl get configmaps -n astroml
    
    local configmaps=(
        "astroml-config"
        "feature-store-config"
        "postgres-config"
        "prometheus-config"
        "grafana-config"
        "fluentd-config"
    )
    
    local failed_configmaps=0
    
    for configmap in "${configmaps[@]}"; do
        if kubectl get configmap $configmap -n astroml > /dev/null 2>&1; then
            print_status "✓ $configmap exists"
        else
            print_warning "✗ $configmap does not exist"
            failed_configmaps=$((failed_configmaps + 1))
        fi
    done
    
    return $failed_configmaps
}

# Function to verify secrets
verify_secrets() {
    print_header "Verifying Secrets"
    
    print_status "Secrets in astroml namespace:"
    kubectl get secrets -n astroml
    
    local secrets=(
        "postgres-secret"
        "grafana-secret"
    )
    
    local failed_secrets=0
    
    for secret in "${secrets[@]}"; do
        if kubectl get secret $secret -n astroml > /dev/null 2>&1; then
            print_status "✓ $secret exists"
        else
            print_warning "✗ $secret does not exist"
            failed_secrets=$((failed_secrets + 1))
        fi
    done
    
    return $failed_secrets
}

# Function to verify HPA
verify_hpa() {
    print_header "Verifying Horizontal Pod Autoscalers"
    
    if kubectl get hpa -n astroml > /dev/null 2>&1; then
        print_status "HPA resources in astroml namespace:"
        kubectl get hpa -n astroml
        return 0
    else
        print_warning "No HPA resources found"
        return 1
    fi
}

# Function to test connectivity
test_connectivity() {
    print_header "Testing Connectivity"
    
    # Test Feature Store
    print_status "Testing Feature Store connectivity..."
    if kubectl exec -n astroml deployment/feature-store -- python -c "
from astroml.features import create_feature_store
store = create_feature_store('/app/feature_store')
print('Feature Store is accessible')
" 2>/dev/null; then
        print_status "✓ Feature Store is accessible"
    else
        print_warning "✗ Feature Store connectivity test failed"
    fi
    
    # Test PostgreSQL
    print_status "Testing PostgreSQL connectivity..."
    if kubectl exec -n astroml deployment/postgres -- pg_isready -U astroml > /dev/null 2>&1; then
        print_status "✓ PostgreSQL is accessible"
    else
        print_warning "✗ PostgreSQL connectivity test failed"
    fi
    
    # Test Redis
    print_status "Testing Redis connectivity..."
    if kubectl exec -n astroml deployment/redis -- redis-cli ping | grep -q "PONG"; then
        print_status "✓ Redis is accessible"
    else
        print_warning "✗ Redis connectivity test failed"
    fi
}

# Function to check resource usage
check_resource_usage() {
    print_header "Checking Resource Usage"
    
    print_status "Pod resource usage:"
    kubectl top pods -n astroml 2>/dev/null || print_warning "Metrics server not available"
    
    print_status "Node resource usage:"
    kubectl top nodes 2>/dev/null || print_warning "Metrics server not available"
}

# Function to generate report
generate_report() {
    print_header "Verification Report"
    
    echo "Kubernetes Deployment Verification completed on $(date)"
    echo "======================================================"
    echo ""
    echo "Components Verified:"
    echo "- Namespace"
    echo "- Deployments"
    echo "- Pods"
    echo "- Services"
    echo "- Ingress"
    echo "- Persistent Volumes"
    echo "- ConfigMaps"
    echo "- Secrets"
    echo "- Horizontal Pod Autoscalers"
    echo "- Connectivity"
    echo "- Resource Usage"
    echo ""
    echo "For detailed information, check the output above."
    echo ""
    echo "Next Steps:"
    echo "1. Review any warnings or errors above"
    echo "2. Check logs for failed components: kubectl logs <pod-name> -n astroml"
    echo "3. Access services: kubectl port-forward -n astroml svc/<service-name> <port>:<port>"
    echo "4. Monitor deployment: kubectl get pods -n astroml -w"
}

# Main execution
main() {
    print_header "AstroML Kubernetes Deployment Verification"
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    local failed_checks=0
    
    # Run verification steps
    check_prerequisites || ((failed_checks++))
    verify_namespace || ((failed_checks++))
    verify_deployments || ((failed_checks++))
    verify_pods || ((failed_checks++))
    verify_services || ((failed_checks++))
    verify_ingress || ((failed_checks++))
    verify_persistent_volumes || ((failed_checks++))
    verify_configmaps || ((failed_checks++))
    verify_secrets || ((failed_checks++))
    verify_hpa || ((failed_checks++))
    test_connectivity
    check_resource_usage
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    if [ $failed_checks -eq 0 ]; then
        print_status "✅ All verification checks passed!"
        exit 0
    else
        print_error "❌ $failed_checks verification checks failed"
        exit 1
    fi
}

# Handle signals gracefully
trap 'print_warning "Verification interrupted"; exit 1' SIGINT SIGTERM

# Execute main function
main "$@"
