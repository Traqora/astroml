# Configuration Reference

This document summarizes the configuration files used by AstroML and the validation rules that apply when they are loaded.

## Configuration Layout

- Runtime database settings live in [config/database.yaml](../config/database.yaml).
- Feature-store defaults live in [config/feature_store.yaml](../config/feature_store.yaml).
- Hydra experiment configuration lives under [configs/](../configs/).

## Configuration Precedence

For the configuration values used by AstroML, the effective value is resolved in this order:

1. Explicit runtime overrides such as environment variables or CLI overrides
2. Values from the YAML configuration file
3. Built-in defaults in code

For database connections, the environment variable `ASTROML_DATABASE_URL` overrides the value from [config/database.yaml](../config/database.yaml) when it is set.

## Database Configuration

The database settings are read from [config/database.yaml](../config/database.yaml).

### Required vs. optional

| Key | Required | Type | Notes |
| --- | --- | --- | --- |
| `database.host` | No | string | Non-empty host name or IP |
| `database.port` | No | integer | Valid TCP port, usually `5432` |
| `database.name` | No | string | Database name |
| `database.user` | No | string | Database user name |
| `database.password` | No | string | Keep this secret and avoid committing it |
| `database.pool.*` | No | integer / boolean | Controls connection pooling behavior |

### Common examples

```yaml
database:
  host: localhost
  port: 5432
  name: astroml
  user: astroml
  password: ""
```

### Validation logic

Database settings are validated at load time. Empty host names, invalid ports, and malformed values cause fast failures before the application tries to open a connection.

## Feature Store Configuration

The feature store defaults are configured in [config/feature_store.yaml](../config/feature_store.yaml).

### Key parameters

| Key | Type | Expected value |
| --- | --- | --- |
| `feature_store.storage_path` | string | Local directory path for the SQLite-backed store |
| `cache.ttl_seconds` | integer | Positive TTL in seconds |
| `cache.maxsize` | integer | Positive cache entry limit |
| `cache.max_size_mb` | integer | Positive memory cap in megabytes |

### Validation logic

The feature store uses these values as defaults. Constructor arguments supplied at runtime override them when present.

## Hydra Configuration Groups

The repository includes Hydra configuration groups under [configs/](../configs/).

### `experiment`

Controls experiment metadata and runtime settings.

| Key | Type | Notes |
| --- | --- | --- |
| `experiment.name` | string | Experiment identifier |
| `experiment.seed` | integer | Reproducibility seed |
| `experiment.device` | string | `auto`, `cpu`, or `cuda` |
| `experiment.save_dir` | string | Output directory |
| `experiment.log_level` | string | Logging verbosity |

### `training`

Controls optimization and training behavior.

| Key | Type | Notes |
| --- | --- | --- |
| `epochs` | integer | Positive number of epochs |
| `lr` | float | Positive learning rate |
| `weight_decay` | float | Non-negative regularization weight |
| `batch_size` | integer or null | Optional batch size |
| `val_split` | float | Fraction between `0.0` and `1.0` |
| `test_split` | float | Fraction between `0.0` and `1.0` |
| `temporal_split.enabled` | boolean | Enables time-ordered splits |

### `sampling`

Controls graph sampling behavior.

| Key | Type | Notes |
| --- | --- | --- |
| `fanout` | list of integers | Positive fanout values for sampling layers |
| `batch_size` | integer | Positive batch size for sampling |

### `database`

Database settings are not currently exposed as a Hydra group. They are loaded from [config/database.yaml](../config/database.yaml) and can be overridden with `ASTROML_DATABASE_URL`.

## Common Configuration Examples

### Local development

```yaml
# config/database.yaml

database:
  host: localhost
  port: 5432
  name: astroml
  user: astroml
  password: ""
```

```yaml
# configs/experiment/inductive.yaml
experiment:
  name: local_dev
  seed: 42
  device: cpu
```

### CPU training

```yaml
training:
  epochs: 50
  lr: 0.001
  batch_size: 32
```

### Graph sampling

```yaml
sampling:
  fanout: [25, 10]
  batch_size: 512
```

## Configuration Schema Diagram

```text
Runtime overrides
  └─ YAML config files
       └─ Built-in defaults

config/database.yaml
config/feature_store.yaml
configs/{experiment,training,sampling,model,data}/...
```

## Validation Guidance

When you change configuration values:

- Keep secrets out of version control.
- Prefer small, explicit changes to existing keys rather than introducing new ones without updating the documentation.
- Validate the YAML structure before running experiments or ingestion jobs.
- Use environment variables for deployment-specific values such as database credentials.
