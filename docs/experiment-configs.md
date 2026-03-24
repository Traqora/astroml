# Experiment Configuration Guide

This guide explains how to use the Hydra configuration system to run ML experiments with AstroML.

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default configuration
python train.py

# Override learning rate
python train.py training.lr=0.001

# Use debug experiment
python train.py experiment=debug

# Override multiple parameters
python train.py model.hidden_dims=[128,64] training.lr=0.01 training.epochs=300
```

### Hyperparameter Sweeps

```bash
# Grid search over learning rates
python train.py --multirun training.lr=0.001,0.01,0.1

# Grid search over multiple parameters
python train.py --multirun model.hidden_dims=[32],[64,32],[128,64] training.lr=0.001,0.01

# Use pre-configured sweep
python train.py --config-name experiments/hyperparameter_search --multirun
```

## 📁 Configuration Structure

```
configs/
├── config.yaml                    # Main configuration file
├── model/
│   └── gcn.yaml                  # GCN model configurations
├── training/
│   └── default.yaml              # Training configurations
├── data/
│   └── cora.yaml                 # Dataset configurations
└── experiments/
    ├── debug.yaml                # Debug experiment
    ├── baseline.yaml             # Baseline experiment
    └── hyperparameter_search.yaml # Hyperparameter sweep
```

## ⚙️ Configuration Files

### Main Config (`configs/config.yaml`)

The main configuration file that sets up defaults and experiment settings:

```yaml
defaults:
  - model: gcn          # Use GCN model
  - training: default   # Use default training config
  - data: cora         # Use Cora dataset
  - _self_             # Include this file's settings

experiment:
  name: "astroml_experiment"
  seed: 42
  device: "auto"
  save_dir: "outputs"
  log_level: "INFO"
```

### Model Config (`configs/model/gcn.yaml`)

Configures the Graph Convolutional Network:

```yaml
_target_: astroml.models.gcn.GCN
input_dim: ???           # Will be set from dataset
hidden_dims: [64, 32]   # Hidden layer sizes
output_dim: ???          # Will be set from dataset
dropout: 0.5
activation: "relu"
batch_norm: false
residual: false
```

### Training Config (`configs/training/default.yaml`)

Training hyperparameters and settings:

```yaml
epochs: 200
lr: 0.01
weight_decay: 5e-4
optimizer: "adam"
scheduler: null
early_stopping:
  patience: 50
  min_delta: 1e-4
  monitor: "val_loss"
  mode: "min"
```

### Data Config (`configs/data/cora.yaml`)

Dataset configuration:

```yaml
_target_: torch_geometric.datasets.Planetoid
name: "Cora"
root: "data"
transform:
  _target_: torch_geometric.transforms.NormalizeFeatures
```

## 🔧 Configuration Overrides

### Command Line Overrides

You can override any configuration parameter from the command line:

```bash
# Override learning rate
python train.py training.lr=0.001

# Override model architecture
python train.py model.hidden_dims=[128,64,32] model.dropout=0.6

# Override dataset
python train.py data.name=CiteSeer

# Override experiment settings
python train.py experiment.name=my_experiment experiment.seed=123
```

### Using Experiments

Pre-configured experiments provide complete setups:

```bash
# Debug experiment (small model, few epochs)
python train.py --config-name experiments/debug

# Baseline experiment (standard settings)
python train.py --config-name experiments/baseline

# Hyperparameter search experiment
python train.py --config-name experiments/hyperparameter_search --multirun
```

## 📊 Hyperparameter Sweeps

### Basic Grid Search

```bash
# Search over learning rates
python train.py --multirun training.lr=0.001,0.01,0.1

# Search over model architectures
python train.py --multirun model.hidden_dims=[32],[64,32],[128,64]

# Combined search
python train.py --multirun training.lr=0.001,0.01 model.dropout=0.3,0.5,0.7
```

### Using Sweep Configurations

The `hyperparameter_search.yaml` experiment defines a parameter grid:

```bash
python train.py --config-name experiments/hyperparameter_search --multirun
```

This will run experiments for all combinations of:
- `model.hidden_dims`: [32], [64,32], [128,64]
- `model.dropout`: 0.2, 0.5, 0.7
- `training.lr`: 0.001, 0.01, 0.1
- `training.weight_decay`: 1e-5, 5e-4, 1e-3

## 📁 Output Structure

Hydra automatically organizes experiment outputs:

```
outputs/
└── astroml_experiment/
    └── 2024-03-24_10-30-45/
        ├── .hydra/
        │   ├── config.yaml      # Full configuration
        │   ├── hydra.yaml       # Hydra settings
        │   └── overrides.yaml    # Command line overrides
        ├── best_model.pth       # Best model checkpoint
        ├── last_model.pth       # Final model checkpoint
        ├── results.yaml         # Training results
        └── train.log           # Training logs
```

For multirun experiments:

```
outputs/
└── astroml_experiment/
    └── multirun/
        ├── model.hidden_dims=32,training.lr=0.001/
        ├── model.hidden_dims=64,training.lr=0.001/
        └── ...
```

## 🎯 Common Use Cases

### 1. Quick Debugging

```bash
# Small model, few epochs for fast iteration
python train.py experiment=debug
```

### 2. Baseline Comparison

```bash
# Run baseline experiment
python train.py --config-name experiments/baseline

# Compare with different learning rate
python train.py --config-name experiments/baseline training.lr=0.001
```

### 3. Architecture Search

```bash
# Test different model sizes
python train.py --multirun model.hidden_dims=[32],[64,32],[128,64,32]
```

### 4. Learning Rate Tuning

```bash
# Fine-grained learning rate search
python train.py --multirun training.lr=0.001,0.003,0.01,0.03,0.1
```

### 5. Regularization Experiments

```bash
# Test different dropout rates
python train.py --multirun model.dropout=0.1,0.3,0.5,0.7

# Test weight decay
python train.py --multirun training.weight_decay=0,1e-5,5e-4,1e-3
```

## 🔍 Advanced Features

### Custom Configurations

Create your own experiment configurations:

```yaml
# configs/experiments/my_experiment.yaml
defaults:
  - override /model: gcn
  - override /training: default
  - override /data: cora

experiment:
  name: "my_custom_experiment"

model:
  hidden_dims: [256, 128]
  dropout: 0.6

training:
  epochs: 500
  lr: 0.003
```

### Environment Variables

Use environment variables in configs:

```yaml
# In config.yaml
experiment:
  name: "${oc.env:USER}_experiment"
  seed: ${oc.env:RANDOM_SEED:42}
```

### Conditional Configuration

Use conditional logic in configs:

```yaml
# Conditional model size based on dataset
model:
  hidden_dims: ${select:${data.name},CiteSeer:[64],PubMed:[128,64],default:[64,32]}
```

## 📝 Best Practices

1. **Use descriptive experiment names** for easy identification
2. **Set random seeds** for reproducible results
3. **Use early stopping** to prevent overfitting
4. **Save both best and last models** for comparison
5. **Log frequently** during training for debugging
6. **Use multirun for systematic hyperparameter searches**
7. **Keep configuration files under version control**

## 🆘 Troubleshooting

### Common Issues

1. **Config not found**: Check file paths and YAML syntax
2. **Override not working**: Use dot notation (e.g., `model.lr` not `lr`)
3. **Multirun not working**: Ensure `--multirun` flag is used
4. **Output directory issues**: Check write permissions

### Debugging Configurations

```bash
# Print configuration without running
python train.py --cfg

# Print specific config section
python train.py --cfg model

# Dry run to check config
python train.py --dry-run
```

## 📚 Additional Resources

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [PyTorch Lightning Integration](https://pytorch-lightning.readthedocs.io/)

---

For more examples and advanced configurations, see the `configs/experiments/` directory.
