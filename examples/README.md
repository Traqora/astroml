# Example Notebooks

This directory contains Jupyter notebooks demonstrating AstroML's core
functionality for Stellar blockchain graph ML.

## Prerequisites

Before running any notebook, install the project and its dependencies:

```bash
# From the repository root
pip install -r requirements.txt
pip install -e .
```

### Kernel Setup

Make sure your Jupyter kernel uses the virtual environment where AstroML
is installed:

```bash
python -m ipykernel install --user --name=astroml --display-name="Python (astroml)"
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_getting_started.ipynb` | End-to-end walkthrough: ingestion → graph → training |
| `02_fraud_detection.ipynb` | Fraud pattern injection, Deep SVDD, and GNN scoring |
| `03_transaction_graph_analysis.ipynb` | Temporal snapshots, structural importance, and feature engineering |

## Verifying Your Setup

Each notebook starts with a dependency-check cell that validates all
required packages are importable. If that cell produces warnings, install
the missing dependencies before proceeding.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'astroml'`** — run `pip install -e .`
  from the repository root, or add the root to `sys.path` (see the first code
  cell of each notebook).
- **Missing `torch` / `torch_geometric`** — install via
  `pip install -r requirements-cpu.txt` (CPU) or follow instructions at
  [pytorch.org](https://pytorch.org) for a CUDA build.
