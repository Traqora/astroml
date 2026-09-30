# Troubleshooting Guide

This guide collects common issues encountered while installing, configuring, ingesting data, training models, or running the API.

## Installation Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| `ModuleNotFoundError` for a dependency | The virtual environment was not created or dependencies were not installed | Create a fresh virtual environment and run `pip install -r requirements.txt` |
| `python` points to the wrong interpreter | Multiple Python versions are installed | Use the interpreter that owns the active virtual environment |
| Import errors after pulling a new branch | Dependencies changed or the package is not installed in editable mode | Reinstall dependencies and run `pip install -e .` |

## Database Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| Connection refused | PostgreSQL is not running or the host/port is incorrect | Confirm the database service is running and update [config/database.yaml](../config/database.yaml) |
| Authentication failed | Wrong username, password, or role | Check credentials and verify the database user has access to the target database |
| `ASTROML_DATABASE_URL` is ignored | The environment variable is not exported in the current shell | Export the variable in the active shell before running the command |

## Ingestion Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| Backfill stops with timeout errors | Network issues or rate limiting from upstream services | Retry with a smaller ledger range and check network connectivity |
| Duplicate rows appear after reruns | Ingestion is not fully idempotent for the target range | Re-run carefully and verify state tracking before repeating the operation |
| Missing data after ingestion | The requested ledger range or filters were incorrect | Validate the requested range and confirm the database configuration |

## Training Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| CUDA out-of-memory errors | The model or batch size is too large for the GPU | Reduce `batch_size`, lower model width, or switch to CPU |
| Training loss does not improve | Hyperparameters may be poorly chosen or the data split is leaking | Review the config under [configs/](../configs/) and confirm the split settings |
| Reproducibility differs across runs | Random seeds or config values changed | Fix the seed and keep the config under version control |

## API Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| CORS errors when calling the API | Browser policy or API configuration mismatch | Verify the allowed origins and confirm the request headers |
| Authentication failures | Missing or invalid credentials | Check the environment variables or headers used by the client |
| Empty or unexpected responses | The service is failing before the request completes | Review the service logs and confirm that the database connection is healthy |

## Collecting Diagnostic Information

When reporting an issue, gather the following:

- The exact command that failed
- The relevant configuration files, especially [config/database.yaml](../config/database.yaml) and [config/feature_store.yaml](../config/feature_store.yaml)
- The traceback or error output
- The Python version and installed package versions
- Any relevant logs from the failed run

## Related Documentation

- [README.md](../README.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [docs/CONFIGURATION.md](./CONFIGURATION.md)
- [SECURITY.md](../SECURITY.md)
