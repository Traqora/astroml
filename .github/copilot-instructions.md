# Copilot / AI agent instructions for AstroML

This file contains concise, actionable guidance to help an AI coding agent be immediately productive in this repository.

## Quick summary
- AstroML is a Python research framework for dynamic graph ML on Stellar. Key workflows are ingestion, graph building, feature extraction, and training (see README).
- Useful entry points: the command-line modules invoked in README such as `astroml.ingestion.backfill`, `astroml.graph.build_snapshot`, and `astroml.training.train_gcn`.

## Discovery checklist (run first)
- List top-level files: `ls -la` and `tree -L 2`.
- Find build/test files: `git grep -n "requirements.txt\|pyproject.toml\|setup.py\|Makefile\|Dockerfile" || true`.
- Find Python entrypoints and modules: `git grep -n "^if __name__ == '__main__'\|def main\|module" || true`.
- Search for keywords used in README: `git grep -n "backfill\|build_snapshot\|train_gcn\|config/database.yaml"`.

## Project-specific patterns & conventions
- CLI modules: Many workflows are exposed as `python -m astroml.<submodule>`. Prefer running those modules when reproducing workflows (examples are in README).
- Config: Database and environment configuration live under `config/` (README references `config/database.yaml`). Update/read these before running ingestion or tests.
- Data flow: Ledger → Ingestion → Normalization → Graph Builder → Features → Models. When adding a change, map it to one of these stages.

## Common commands (from README)
- Create environment and install: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Backfill ledger ingestion: `python -m astroml.ingestion.backfill --start-ledger 1000000 --end-ledger 1100000`
- Build graph snapshot: `python -m astroml.graph.build_snapshot --window 30d`
- Train baseline GCN: `python -m astroml.training.train_gcn`

## Where to look for help in code reviews
- When reviewing changes that affect ingestion, ensure database config and idempotency (re-run backfills safely).
- For graph building changes, check windowing logic and reproducibility (random seeds, snapshot metadata).
- For model/training changes, verify experiment reproducibility (config-driven, checkpoint paths) and alignment with README examples.

## Integration & runtime hints
- Expect PostgreSQL for ingestion; tests or local runs may require a local Postgres or a test fixture.
- PyTorch / PyG are used for models — GPU availability changes performance but not required for basic runs.

## When you can't find answers in the repo
- If modules, tests, or config files are missing, run the discovery checklist above and ask the maintainer for missing artifacts (tests, `requirements.txt`, CI workflows).

## Quick examples for the agent
- To run the ingestion example locally: ensure `config/database.yaml` points to a local Postgres, then run the backfill command from the README.
- To run a targeted search for tests: `git grep -n "pytest\|unittest" || true`.

---

