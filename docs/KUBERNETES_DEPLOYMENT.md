# Kubernetes Deployment Guide for AstroML

This guide provides comprehensive instructions for deploying AstroML with Feature Store to Kubernetes clusters.

## Overview

The Kubernetes deployment provides:
- **Scalable deployment** with horizontal pod autoscaling
- **High availability** with multiple replicas
- **Monitoring** with Prometheus and Grafana
- **Logging** with Elasticsearch, Fluentd, and Kibana (EFK stack)
- **Ingress** for external access
- **CI/CD pipeline** with GitHub Actions

## Prerequisites

### System Requirements
- **Kubernetes cluster** v1.24+ (EKS, GKE, AKS, or minikube)
- **kubectl** v1.24+ configured for cluster access
- **kustomize** v4.0+ for configuration management
- **Helm** v3.0+ (optional, for additional packages)
- **Storage class** configured for persistent volumes
- **Ingress controller** installed (nginx, traefik, etc.)

### Installation

#### kubectl
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify installation
kubectl version --client
```

#### kustomize
```bash
# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify installation
kustomize version
```

## Deployment Architecture

### Components

#### Core Infrastructure
- **PostgreSQL** - Primary database with persistent storage
- **Redis** - Caching and job queues
- **Feature Store** - Dedicated feature management service

#### Application Services
- **Ingestion Service** - Data processing and backfill
- **Training Service** - ML model training
- **API Service** - REST API for feature access

#### Monitoring Stack
- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization and dashboards

#### Logging Stack
- **Elasticsearch** - Log storage and search
- **Fluentd** - Log collection and aggregation
- **Kibana** - Log visualization and analysis

### Network Architecture

```
Internet
    ↓
Ingress Controller
    ↓
AstroML Services
    ↓
Feature Store, Ingestion, Training
    ↓
PostgreSQL, Redis
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Menjay7/astroml.git
cd astroml
```

### 2. Configure Secrets
```bash
# Create secrets file
cat > k8s/secrets.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: astroml
type: Opaque
stringData:
  password: your-secure-password-here
---
apiVersion: v1
kind: Secret
metadata:
  name: astroml-secret
  namespace: astroml
type: Opaque
stringData:
  database-url: "postgresql://astroml:your-password@postgres:5432/astroml"
  redis-url: "redis://redis:6379/0"
EOF
```

### 3. Deploy Using Script
```bash
# Make script executable
chmod +x scripts/deploy-k8s.sh

# Deploy all components
./scripts/deploy-k8s.sh deploy
```

### 4. Verify Deployment
```bash
# Check pod status
kubectl get pods -n astroml

# Check services
kubectl get services -n astroml

# Check ingress
kubectl get ingress -n astroml
```

### 5. Access Services
```bash
# Access Grafana
kubectl port-forward -n astroml svc/grafana 3000:3000
# Open browser: http://localhost:3000 (admin/admin)

# Access Kibana
kubectl port-forward -n astroml svc/kibana 5601:5601
# Open browser: http://localhost:5601
```

## Deployment Methods

### Method 1: Using Deployment Script

```bash
# Deploy all components
./scripts/deploy-k8s.sh deploy

# Deploy using kustomize
./scripts/deploy-k8s.sh kustomize

# Deploy monitoring only
./scripts/deploy-k8s.sh monitoring

# Deploy logging only
./scripts/deploy-k8s.sh logging
```

### Method 2: Using kubectl Directly

```bash
# Apply all configurations
kubectl apply -f k8s/

# Apply specific components
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/feature-store-deployment.yaml
```

### Method 3: Using Kustomize

```bash
# Build and apply
kustomize build k8s/ | kubectl apply -f -

# Build and preview
kustomize build k8s/

# Build to file
kustomize build k8s/ > deployment.yaml
kubectl apply -f deployment.yaml
```

## Configuration Management

### Environment-Specific Configurations

Create overlays for different environments:

```bash
# Production overlay
k8s/overlays/production/
├── kustomization.yaml
├── postgres-patch.yaml
└── feature-store-patch.yaml

# Staging overlay
k8s/overlays/staging/
├── kustomization.yaml
├── postgres-patch.yaml
└── feature-store-patch.yaml
```

### Example Production Overlay

```yaml
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: astroml

bases:
  - ../../

patchesStrategicMerge:
  - postgres-patch.yaml
  - feature-store-patch.yaml

images:
  - name: astroml
    newTag: v1.0.0
```

### Example Patch

```yaml
# k8s/overlays/production/postgres-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 3
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "2000m"
```

## Scaling and High Availability

### Horizontal Pod Autoscaling

The Feature Store deployment includes HPA configuration:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: feature-store-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: feature-store
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment/feature-store -n astroml --replicas=5

# Scale using script
./scripts/deploy-k8s.sh scale feature-store 5
```

### Resource Limits

Configure resource limits based on workload:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus metrics:

```bash
# Port forward to Prometheus
kubectl port-forward -n astroml svc/prometheus 9090:9090

# Access in browser
# http://localhost:9090
```

### Grafana Dashboards

Access Grafana for visualization:

```bash
# Port forward to Grafana
kubectl port-forward -n astroml svc/grafana 3000:3000

# Access in browser
# http://localhost:3000
# Default credentials: admin/admin
```

### Log Analysis with Kibana

Access Kibana for log analysis:

```bash
# Port forward to Kibana
kubectl port-forward -n astroml svc/kibana 5601:5601

# Access in browser
# http://localhost:5601
```

## Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n astroml

# Check logs
kubectl logs <pod-name> -n astroml

# Check events
kubectl get events -n astroml --sort-by='.lastTimestamp'
```

#### Service Not Accessible
```bash
# Check service endpoints
kubectl get endpoints <service-name> -n astroml

# Check service configuration
kubectl describe service <service-name> -n astroml

# Check network policies
kubectl get networkpolicies -n astroml
```

#### Storage Issues
```bash
# Check PVC status
kubectl get pvc -n astroml

# Check storage class
kubectl get storageclass

# Check PV status
kubectl get pv
```

### Debugging Commands

```bash
# Get all resources
kubectl get all -n astroml

# Get detailed information
kubectl describe deployment/feature-store -n astroml

# Get logs from all pods
kubectl logs -l app=feature-store -n astroml --all-containers=true

# Execute into pod
kubectl exec -it <pod-name> -n astroml -- /bin/bash

# Check resource usage
kubectl top pods -n astroml
kubectl top nodes
```

## CI/CD Pipeline

### GitHub Actions Workflow

The project includes a comprehensive CI/CD pipeline:

```yaml
# .github/workflows/docker-ci-cd.yml
- Build and test
- Build Docker images
- Security scanning
- Deploy to Kubernetes
- Notification
```

### Pipeline Stages

1. **Build and Test** - Run tests and coverage
2. **Build Docker Images** - Build multi-stage images
3. **Security Scan** - Trivy vulnerability scanning
4. **Deploy to Kubernetes** - Automatic deployment
5. **Notification** - Slack notifications

### Manual Deployment

```bash
# Trigger deployment manually
gh workflow run docker-ci-cd.yml

# Deploy specific branch
gh workflow run docker-ci-cd.yml -f branch=develop
```

## Security Considerations

### Secrets Management

Use Kubernetes secrets for sensitive data:

```bash
# Create secret from file
kubectl create secret generic db-secret \
  --from-literal=password=your-password \
  -n astroml

# Create secret from file
kubectl create secret generic tls-secret \
  --from-file=tls.crt=./cert.pem \
  --from-file=tls.key=./key.pem \
  -n astroml
```

### Network Policies

Implement network policies for security:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: feature-store-network-policy
  namespace: astroml
spec:
  podSelector:
    matchLabels:
      app: feature-store
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: astroml-ingestion
    ports:
    - protocol: TCP
      port: 8000
```

### RBAC Configuration

The deployment includes RBAC configuration:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: astroml
  namespace: astroml
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: astroml-role
  namespace: astroml
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
```

## Backup and Recovery

### Database Backup

```bash
# Backup PostgreSQL
kubectl exec -n astroml postgres-0 -- pg_dump -U astroml astroml > backup.sql

# Restore PostgreSQL
kubectl exec -i -n astroml postgres-0 -- psql -U astroml astroml < backup.sql
```

### Volume Backup

```bash
# Backup persistent volumes
kubectl get pvc -n astroml
# Use your cloud provider's backup solution
```

### Disaster Recovery

```bash
# Restore from backup
kubectl apply -f k8s/
kubectl exec -i -n astroml postgres-0 -- psql -U astroml astroml < backup.sql
```

## Performance Optimization

### Resource Tuning

Adjust resource limits based on usage:

```bash
# Monitor resource usage
kubectl top pods -n astroml

# Update resource limits
kubectl set resources deployment/feature-store \
  -n astroml \
  --limits=cpu=2000m,memory=2Gi \
  --requests=cpu=1000m,memory=1Gi
```

### Caching Configuration

Optimize Redis caching:

```yaml
env:
- name: FEATURE_STORE_CACHE_SIZE
  value: "5000"
- name: FEATURE_STORE_CACHE_TTL
  value: "7200"
```

### Database Optimization

Configure PostgreSQL for performance:

```yaml
env:
- name: POSTGRES_SHARED_BUFFERS
  value: "256MB"
- name: POSTGRES_EFFECTIVE_CACHE_SIZE
  value: "1GB"
```

## Maintenance

### Rolling Updates

```bash
# Update deployment
kubectl set image deployment/feature-store \
  feature-store=astroml:latest \
  -n astroml

# Rollout status
kubectl rollout status deployment/feature-store -n astroml

# Rollback if needed
kubectl rollout undo deployment/feature-store -n astroml
```

### Cleanup

```bash
# Remove all components
./scripts/deploy-k8s.sh cleanup

# Remove specific components
kubectl delete -f k8s/feature-store-deployment.yaml -n astroml

# Remove namespace
kubectl delete namespace astroml
```

## Best Practices

1. **Always use secrets** for sensitive data
2. **Implement resource limits** to prevent resource exhaustion
3. **Use liveness and readiness probes** for health checks
4. **Implement network policies** for security
5. **Monitor resource usage** regularly
6. **Backup data regularly**
7. **Test deployments in staging first**
8. **Use version tags** for images
9. **Implement proper RBAC** for access control
10. **Document custom configurations**

## Support

For issues and questions:
1. Check this documentation
2. Review logs and error messages
3. Search GitHub issues
4. Create new issue with details

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Elastic Stack Documentation](https://www.elastic.co/guide/)
