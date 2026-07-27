from __future__ import annotations

import argparse
import json



def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astroml",
        description=CLI_DESCRIPTION,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the database YAML config (default: config/database.yaml). "
            "Used by `config --print-db` and any subcommand that reads the "
            "database configuration."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Runtime environment name (e.g., development, production). "
            "When provided, sets ASTROML_ENV for downstream loaders unless "
            "ASTROML_ENV is already set in the process environment."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Incremental ingestion of ledgers")
    ingest.add_argument("--start", type=int, default=None, help="Start ledger id (inclusive)")
    ingest.add_argument("--end", type=int, default=None, help="End ledger id (inclusive)")
    ingest.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to state file (defaults to ./.astroml_state/ingestion_state.json)",
    )

    config = sub.add_parser("config", help="Configuration management")
    config.add_argument(
        "--print-db",
        action="store_true",
        help="Print effective database configuration",
    )

    quickstart = sub.add_parser(
        "quickstart",
        help="Run quick start: ingestion → graph → train pipeline with sample data",
    )
    quickstart.add_argument(
        "--num-ledgers",
        type=int,
        default=100,
        help="Number of sample ledgers to generate (default: 100)",
    )
    quickstart.add_argument(
        "--num-accounts",
        type=int,
        default=50,
        help="Number of sample accounts (default: 50)",
    )
    quickstart.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10)",
    )
    quickstart.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    preprocess = sub.add_parser(
        "preprocess-backfill",
        help="Preprocess large ledger backfill datasets using Polars",
    )
    preprocess.add_argument(
        "--input",
        required=True,
        help="Input file or directory (csv, parquet, ndjson/jsonl).",
    )
    preprocess.add_argument(
        "--output",
        required=True,
        help="Output Parquet path.",
    )
    preprocess.add_argument(
        "--input-format",
        choices=["parquet", "csv", "ndjson", "jsonl"],
        default=None,
        help="Optional explicit input format.",
    )


    if args.command == "ingest":
        store = StateStore(path=args.state_file) if args.state_file else StateStore()
        service = IngestionService(state_store=store)

        # Example fetch/process functions; in real usage, users would customize/import
        def fetch_fn(ledger_id: int):
            # Placeholder fetch, replace with real data retrieval
            return {"ledger": ledger_id, "data": f"payload-{ledger_id}"}

        def process_fn(ledger_id: int, payload: dict):
            # Placeholder processing; replace with DB writes or other side effects
            # For CLI visibility we do minimal printing; real apps would use logging
            print(f"processed ledger {ledger_id}")

        result = service.ingest(
            start_ledger=args.start,
            end_ledger=args.end,
            fetch_fn=fetch_fn,
            process_fn=process_fn,
        )
        print(json.dumps({
            "attempted": result.attempted,
            "processed": result.processed,
            "skipped": result.skipped,
        }, indent=2))
        return 0

    if args.command == "config":
        if args.print_db:
            try:
                db_config = load_database_config(args.config)
                print("Effective database configuration:")
                print(json.dumps({
                    "host": db_config.host,
                    "port": db_config.port,
                    "name": db_config.name,
                    "user": db_config.user,
                    "password": "***" if db_config.password else "",
                    "url": db_config.to_url()
                }, indent=2))
                return 0
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except Exception as e:
                print(f"Error loading config: {e}")
                return 1
        else:
            config.print_help()
            return 1

    if args.command == "quickstart":
        from .quick_start import run_quickstart, QuickStartConfig
        
        # Update config with CLI arguments
        QuickStartConfig.NUM_SAMPLE_LEDGERS = args.num_ledgers
        QuickStartConfig.NUM_ACCOUNTS = args.num_accounts
        QuickStartConfig.TRAIN_EPOCHS = args.epochs
        QuickStartConfig.RANDOM_SEED = args.seed
        
        return run_quickstart()

    if args.command == "preprocess-backfill":
        from .preprocessing.ledger_backfill import preprocess_to_parquet

        output_path = preprocess_to_parquet(
            input_path=args.input,
            output_path=args.output,
            input_format=args.input_format,
        )
        print(json.dumps({"output": str(output_path)}, indent=2))
        return 0
    
    if args.command == "models":
        db = _sync_session_factory()()
        
        if args.subcommand == "register" or args.subcommand == "version":
            name = args.name if args.subcommand == "register" else args.model_name
            metrics = json.loads(args.metrics) if args.metrics else None
            existing = db.scalar(
                select(ModelRegistry).where(
                    ModelRegistry.name == name, ModelRegistry.version == args.version
                )
            )
            if existing:
                print(f"Error: Model '{name}' version '{args.version}' already exists")
                return 1
            
            entry = ModelRegistry(
                name=name,
                version=args.version,
                path=args.path,
                owner=args.owner,
                tags=args.tags,
                mlflow_run_id=args.mlflow_run_id,
                metrics=metrics,
                status=args.status,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            print(json.dumps({
                "id": entry.id,
                "name": entry.name,
                "version": entry.version,
                "path": entry.path,
                "owner": entry.owner,
                "tags": entry.tags,
                "mlflow_run_id": entry.mlflow_run_id,
                "status": entry.status,
                "created_at": entry.created_at.isoformat()
            }, indent=2))
            return 0
        
        elif args.subcommand == "transition":
            entry = db.scalar(
                select(ModelRegistry).where(
                    ModelRegistry.name == args.model_name,
                    ModelRegistry.version == args.version
                )
            )
            if not entry:
                print(f"Error: Model '{args.model_name}' version '{args.version}' not found")
                return 1
            
            if args.stage == "active":
                db.execute(
                    update(ModelRegistry)
                    .where(ModelRegistry.name == args.model_name, ModelRegistry.id != entry.id)
                    .values(status="inactive")
                )
            
            entry.status = args.stage
            db.commit()
            db.refresh(entry)
            print(json.dumps({
                "id": entry.id,
                "name": entry.name,
                "version": entry.version,
                "status": entry.status
            }, indent=2))
            return 0
        
        elif args.subcommand == "list":
            query = select(ModelRegistry)
            if args.name:
                query = query.where(ModelRegistry.name == args.name)
            if args.status:
                query = query.where(ModelRegistry.status == args.status)
            if args.owner:
                query = query.where(ModelRegistry.owner == args.owner)
            if args.tags:
                for tag in args.tags:
                    query = query.where(ModelRegistry.tags.contains([tag]))
            
            # Count total
            count_query = select(func.count()).select_from(query.subquery())
            total = db.scalar(count_query) or 0
            
            # Paginate
            offset = (args.page - 1) * args.page_size
            query = query.order_by(ModelRegistry.created_at.desc()).offset(offset).limit(args.page_size)
            rows = db.scalars(query).all()
            
            result = {
                "page": args.page,
                "page_size": args.page_size,
                "total": total,
                "data": [
                    {
                        "id": row.id,
                        "name": row.name,
                        "version": row.version,
                        "path": row.path,
                        "owner": row.owner,
                        "tags": row.tags,
                        "mlflow_run_id": row.mlflow_run_id,
                        "status": row.status,
                        "created_at": row.created_at.isoformat()
                    }
                    for row in rows
                ]
            }
            print(json.dumps(result, indent=2))
            return 0
        
        elif args.subcommand == "compare":
            v1 = db.scalar(
                select(ModelRegistry).where(
                    ModelRegistry.name == args.model_name,
                    ModelRegistry.version == args.version1
                )
            )
            v2 = db.scalar(
                select(ModelRegistry).where(
                    ModelRegistry.name == args.model_name,
                    ModelRegistry.version == args.version2
                )
            )
            
            if not v1:
                print(f"Error: Model '{args.model_name}' version '{args.version1}' not found")
                return 1
            if not v2:
                print(f"Error: Model '{args.model_name}' version '{args.version2}' not found")
                return 1
            
            comparison = {
                "models": [
                    {
                        "id": v1.id,
                        "name": v1.name,
                        "version": v1.version,
                        "mlflow_run_id": v1.mlflow_run_id,
                        "metrics": v1.metrics
                    },
                    {
                        "id": v2.id,
                        "name": v2.name,
                        "version": v2.version,
                        "mlflow_run_id": v2.mlflow_run_id,
                        "metrics": v2.metrics
                    }
                ],
                "metrics_diff": {}
            }
            
            # Show metric differences
            all_metrics = set()
            if v1.metrics:
                all_metrics.update(v1.metrics.keys())
            if v2.metrics:
                all_metrics.update(v2.metrics.keys())
            
            for key in all_metrics:
                val1 = v1.metrics.get(key) if v1.metrics else None
                val2 = v2.metrics.get(key) if v2.metrics else None
                comparison["metrics_diff"][key] = {"version1": val1, "version2": val2}
            
            print(json.dumps(comparison, indent=2))
            return 0
        
        elif args.subcommand == "load-metadata":
            # Use MLflowTracker to load metadata
            try:
                from .tracking.mlflow_tracker import MLflowTracker
                tracker = MLflowTracker(enabled=True)
                metadata = tracker.load_run_metadata(args.model_name, args.version)
                if metadata:
                    print(json.dumps(metadata, indent=2, default=str))
                    return 0
                else:
                    print("No metadata found")
                    return 1
            except ImportError as e:
                print(f"Error: MLflow not available: {e}")
                return 1

    if args.command == "llm":
        if hasattr(args, "func"):
            args.func(args)
            return 0
        llm_parser.print_help()
        return 1

    if args.command == "agent":
        from .agent.config import AgentConfig, LLMConfig, MemoryConfig, ExecutorConfig
        from .agent.executor import AutonomousExecutor
        from .agent.tools import create_default_registry

        if args.agent_command == "tools":
            registry = create_default_registry()
            print(json.dumps({"tools": sorted(registry.list_tools())}, indent=2))
            return 0

        # agent run
        llm_config = LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
        )
        agent_config = AgentConfig(
            llm=llm_config,
            agent_type=args.agent_type,
            verbose=args.verbose,
        )
        executor = AutonomousExecutor(agent_config=agent_config)
        result = executor.run(task_description=args.task or "")

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        elif args.verbose:
            print(json.dumps({
                "task": result.task,
                "success": result.success,
                "output": result.output,
                "elapsed_seconds": result.elapsed_seconds,
                "steps": len(result.steps),
                "error": result.error,
            }, indent=2))
        else:
            print(result.output)

        return 0 if result.success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
