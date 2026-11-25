# V1 Operations

## Snapshot build reporting

Each successful database-backed `SnapshotWindow` exposes `build_duration_s`.
The snapshot logger emits structured fields `window_index`,
`duration_seconds`, `node_count`, and `edge_count`. These operational values
are intentionally not part of deterministic snapshot lineage. The graph CLI
also prints duration and graph size in its normal summary; `--memory-profile`
adds memory measurements. The synthetic CLI mode works without a database.

## Ingestion retry classification

`AstroMLError.retryable` controls retries for application-level ingestion
errors. `SourceUnavailableError` is retryable; `ParseError` and base
`IngestionError` are permanent by default. The enhanced Horizon stream raises
permanent errors without another request or backoff. Transport connection
errors and rate limits retain their dedicated bounded retry policies.

## File path preflight

The synthetic fraud injector validates its input, output, and optional summary
paths before loading or transforming transactions. The input must be a
readable regular file; destinations must not be directories, their nearest
existing parent must be writable, and all supplied paths must resolve to
different files. Missing output parent directories are created only after all
paths pass validation. Invalid paths fail before the source file is modified.

## Startup model loading

Set `MODEL_ARTIFACT_PATH` to a trusted local model artifact to load it once
during API lifespan startup. `MODEL_NAME` selects its in-process name and
defaults to `default`. Missing or unreadable artifacts fail startup before
the batch scheduler is started. The default loader deserializes trusted
Python pickle artifacts; provide a custom `ModelLoader(artifact_loader=...)`
for other formats or safer deserialization. Models remain cached by name and
resolved path for the process lifetime; requesting the same name from a
different path is an error.