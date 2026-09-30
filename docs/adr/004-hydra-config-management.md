# ADR-004: Hydra for Configuration Management

## Status

Accepted

## Context

Graph machine learning models, feature extraction pipelines, and benchmarking tasks require complex configuration hierarchies (e.g. data loader parameters, GNN layer counts, learning rates, temporal cutoff windows). Configurations need to be modular, composition-friendly, versionable via YAML, and easily overridden via CLI arguments for hyperparameter sweeps and experimentation.

Alternatives considered:
- **Argparse / Click**: Simple CLI parsing, but cumbersome for multi-level nested parameter hierarchies.
- **Pure YAML / JSON files**: Good for static config, but lacks dynamic composition, variable interpolation, and CLI override support.
- **Python dataclasses / Argparse hybrid**: Lacks clean config merging and multi-experiment file structuring.

## Decision

We chose **Hydra (OmegaConf)** as the framework for configuration management across AstroML training, benchmarking, and experiment pipelines.

Key reasons:
- Modular hierarchical YAML configuration files (`configs/training/`, `configs/model/`).
- Dynamic runtime composition and CLI parameter overrides (`python train.py training.device=cuda training.batch_size=512`).
- Automatic logging of resolved configurations alongside experiment benchmark artifacts.
- Strong Python type safety through OmegaConf structured configs.

## Consequences

### Positive
- Flexible experiment configuration composition without code modification.
- Clean CLI override syntax for parameter sweeps and CI test matrix steps.
- Automatic persistence of exact run configurations in `benchmark_results/`.

### Negative / Tradeoffs
- Requires developers to structure configuration files following Hydra conventions.
- Small framework overhead when initializing the Hydra configuration engine.
