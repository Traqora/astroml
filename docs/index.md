# AstroML Documentation

Welcome to the AstroML documentation!

## 🚀 Quick Start

AstroML is a comprehensive machine learning framework for the Stellar network, providing tools for:

- **Graph Machine Learning**: Advanced GNN models for transaction analysis
- **Fraud Detection**: Sophisticated algorithms for identifying suspicious activity
- **Feature Engineering**: Comprehensive feature extraction and processing
- **Data Ingestion**: Real-time Stellar ledger data processing

## 📚 Documentation Sections

### Machine Learning
- [Graph Construction Architecture](graph-construction.rst)
- [Graph Batch Processing](graph-batch-processing.md)
- [Feature Store](FEATURE_STORE.md)
- [Model Registry](model-registry.md)
- [Model Interpretability](model-interpretability.md)
- [Explainability Reports](explainability-reports.md)
- [Data Quality Validation](DATA_QUALITY_VALIDATION.md)

### Autonomous Agents
- [LLM Agent Framework](agent-framework.md)

### Configuration & Experiments
- [Configuration Reference](CONFIGURATION.md)
- [Experiment Configuration](experiment-configs.md)
- [Hydra Config Management (ADR 004)](adr/004-hydra-config-management.md)

### Performance & Scaling
- [Scaling and Performance Optimization](scaling-optimization.md)
- [Benchmarking Suite](benchmarking.md)
- [Performance Guide](PERFORMANCE.md)
- [Database Query Profiling](database-query-profiling.md)

### Deployment
- [Docker Deployment](docker-deployment.md)
- [Docker Setup](DOCKER_SETUP.md)
- [Kubernetes Deployment](KUBERNETES_DEPLOYMENT.md)
- [Soroban Contract Integration](FRAUD_REGISTRY_CONTRACT.md)
- [GitOps Workflow](gitops-workflow.md)

### Operations
- [Alerting](ALERTING.md)
- [Health Checks](HEALTH_CHECKS.md)
- [Metrics Reference](METRICS_REFERENCE.md)
- [Ingestion Monitoring](ingestion-monitoring.md)
- [PR Size Limits & Exemption Policy](PR_SIZE_LIMITS.md)
- [Runbooks](runbooks/)

### API Reference
- [API Overview](api/index.md)
- [Models API](api/models.md)
- [Ingestion API](api/ingestion.md)
- [Temporal Models API](api/temporal-models.md)
- [Usage Examples](api/usage-examples.md)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/Traqora/astroml.git
cd astroml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For documentation only
pip install -r docs/requirements.txt
```

## 🎯 Quick Examples

### Running Experiments with Hydra

```bash
# Basic experiment
python train.py

# Override parameters
python train.py training.lr=0.001 model.hidden_dims=[128,64]

# Use pre-configured experiments
python train.py --config-name experiments/debug
python train.py --config-name experiments/baseline
```

### Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# Run specific services
docker-compose up postgres redis
docker-compose up ingestion
```

## 📊 Features

### Machine Learning
- **Graph Neural Networks**: GCN, GraphSAGE, GAT implementations
- **Structural Analysis**: Centrality measures, importance metrics
- **Temporal Modeling**: Time-series analysis for transaction patterns

### Data Processing
- **Real-time Ingestion**: Stellar ledger streaming
- **Feature Engineering**: Automated feature extraction
- **Data Validation**: Quality checks and integrity verification

### Deployment
- **Docker Support**: Multi-stage builds for different environments
- **Configuration Management**: Hydra-based experiment tracking
- **Monitoring**: Comprehensive logging and metrics

## 🔗 Links

- [GitHub Repository](https://github.com/Traqora/astroml)
- [Stellar Network](https://www.stellar.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## 📖 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details, or work through the [First PR walkthrough](ONBOARDING.md).

## 📄 License

This project is licensed under the MIT License.
