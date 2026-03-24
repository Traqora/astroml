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
- [Structural Importance Metrics](structural_importance.md)
- [Transaction Graph Analysis](transaction_graph.md)
- [Feature Engineering Pipeline](feature_pipeline.md)

### Configuration & Experiments
- [Experiment Configuration](experiment-configs.md)
- [Hydra Setup Guide](hydra-setup.md)

### Deployment
- [Docker Deployment](docker-deployment.md)
- [Soroban Contract Integration](soroban-contract.md)

### API Reference
- [Models API](api/models.md)
- [Features API](api/features.md)
- [Training API](api/training.md)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/tecch-wiz/astroml.git
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

- [GitHub Repository](https://github.com/tecch-wiz/astroml)
- [Stellar Network](https://www.stellar.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## 📖 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
