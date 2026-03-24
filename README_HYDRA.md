# Hydra Configuration System - Quick Start

## 🚀 Installation

```bash
# Install pip and create virtual environment
sudo apt install python3-pip python3.12-venv -y
python3 -m venv venv
source venv/bin/activate

# Install minimal requirements for Hydra
pip install -r requirements-minimal.txt
```

## 📋 Usage Examples

### Basic Configuration
```bash
# Use default configuration
python test_config.py

# Override parameters
python test_config.py experiments.training.lr=0.001 experiments.model.hidden_dims=[128,64]
```

### Pre-configured Experiments
```bash
# Debug experiment (small model, 10 epochs)
python test_config.py --config-name experiments/debug

# Baseline experiment (standard settings)
python test_config.py --config-name experiments/baseline
```

### Command-line Overrides
```bash
# Learning rate override
python test_config.py experiments.training.lr=0.001

# Model architecture override
python test_config.py experiments.model.hidden_dims=[32]

# Multiple overrides
python test_config.py experiments.training.lr=0.001 experiments.model.hidden_dims=[128,64] experiments.training.epochs=300
```

## 📁 Configuration Structure

```
configs/
├── config.yaml                    # Main configuration
├── model/gcn.yaml                 # GCN model configs
├── training/default.yaml          # Training configs
├── data/cora.yaml                 # Dataset configs
└── experiments/
    ├── debug.yaml                 # Debug experiment
    ├── baseline.yaml              # Baseline experiment
    └── hyperparameter_search.yaml # Parameter sweep
```

## 🎯 Key Features

- **Easy Parameter Override**: `experiments.training.lr=0.001`
- **Pre-configured Experiments**: Debug, baseline, hyperparameter search
- **Automatic Output Organization**: Results saved to timestamped directories
- **Configuration Tracking**: Full config saved with results

## 📊 Test Results

The system successfully demonstrates:
- ✅ Configuration loading and merging
- ✅ Command-line parameter overrides
- ✅ Pre-configured experiment selection
- ✅ Automatic output organization
- ✅ Configuration tracking and logging

## 🔄 Next Steps

1. Install full PyTorch dependencies when needed:
   ```bash
   pip install -r requirements.txt  # May take time due to large downloads
   ```

2. Use the full training script:
   ```bash
   python train.py experiments.training.lr=0.001
   ```

3. Run hyperparameter sweeps:
   ```bash
   python train.py --multirun experiments.training.lr=0.001,0.01,0.1
   ```

---

**The Hydra configuration system is ready for easy experiment management!** 🚀
