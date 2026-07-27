"""LLM CLI command implementations."""

import json
import sys
from typing import Any

from astroml.llm.providers import get_llm_provider
from .config import load_cli_config
from .formatters import (
    print_text,
    print_llm_response,
    print_table,
    print_json,
    print_error,
    output_result,
)
from .interactive import run_chat


def get_provider(provider_name: str = "", model: str = "", **kwargs: Any):
    """Create a provider from CLI args, config, and env."""
    cfg = load_cli_config()
    prov = provider_name or cfg.get("provider", "openai")
    mod = model or cfg.get("model", "")
    return get_llm_provider(prov, model=mod, **kwargs), cfg


def register_llm_subcommands(sub) -> None:
    """Register all LLM subcommands on the argparse subparser group."""

    # ── generate ──
    p = sub.add_parser("generate", help="Generate a completion from a prompt")
    p.add_argument("prompt", nargs="?", default="", help="Prompt text (omit for pipe)")
    p.add_argument("--provider", help="LLM provider (openai, anthropic, etc.)")
    p.add_argument("--model", help="Model name")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max tokens")
    p.add_argument("--temperature", type=float, help="Temperature")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=cmd_generate)

    # ── chat ──
    p = sub.add_parser("chat", help="Interactive chat session with streaming")
    p.add_argument("--provider", help="LLM provider")
    p.add_argument("--model", help="Model name")
    p.add_argument("--system-prompt", default="", help="System prompt")
    p.add_argument("--temperature", type=float, help="Temperature")
    p.set_defaults(func=cmd_chat)

    # ── rag query ──
    p = sub.add_parser("rag", help="RAG operations")
    rag_sub = p.add_subparsers(dest="rag_command", required=True)
    q = rag_sub.add_parser("query", help="Query the RAG pipeline")
    q.add_argument("question", help="Question to ask")
    q.add_argument("--sources", help="Document sources path")
    q.add_argument("--provider", help="LLM provider")
    q.add_argument("--model", help="Model name")
    q.add_argument("--json", action="store_true", help="Output as JSON")
    q.set_defaults(func=cmd_rag_query)

    # ── embed ──
    p = sub.add_parser("embed", help="Generate an embedding vector")
    p.add_argument("text", nargs="?", default="", help="Text to embed")
    p.add_argument("--provider", help="Embedding provider")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=cmd_embed)

    # ── prompts ──
    p = sub.add_parser("prompts", help="Prompt management")
    prompts_sub = p.add_subparsers(dest="prompts_command", required=True)

    prompts_sub.add_parser("list", help="List all prompt templates").set_defaults(func=cmd_prompts_list)

    pr = prompts_sub.add_parser("render", help="Render a prompt template")
    pr.add_argument("name", help="Template name")
    pr.add_argument("--var", action="append", default=[], help="Variable in key=value format")
    pr.set_defaults(func=cmd_prompts_render)

    pt = prompts_sub.add_parser("test", help="Test a prompt with sample data")
    pt.add_argument("name", help="Template name")
    pt.add_argument("--provider", help="LLM provider")
    pt.add_argument("--model", help="Model name")
    pt.add_argument("--var", action="append", default=[], help="Variables")
    pt.set_defaults(func=cmd_prompts_test)

    # ── eval ──
    p = sub.add_parser("eval", help="Evaluation commands")
    eval_sub = p.add_subparsers(dest="eval_command", required=True)

    er = eval_sub.add_parser("run", help="Run an evaluation benchmark")
    er.add_argument("benchmark", help="Benchmark name")
    er.add_argument("--provider", help="LLM provider")
    er.add_argument("--model", help="Model name")
    er.set_defaults(func=cmd_eval_run)

    eval_sub.add_parser("results", help="View evaluation history").set_defaults(func=cmd_eval_results)

    # ── models ──
    sub.add_parser("models", help="List available models").set_defaults(func=cmd_models)

    # ── cost ──
    sub.add_parser("cost", help="Show cost statistics").set_defaults(func=cmd_cost)

    # ── cache ──
    sub.add_parser("cache", help="Show cache statistics").set_defaults(func=cmd_cache)

    # ── backfill ──
    bf = sub.add_parser("backfill", help="Backfill job management")
    bf_sub = bf.add_subparsers(dest="backfill_command", required=True)

    bf_create = bf_sub.add_parser("create", help="Create a new backfill job")
    bf_create.add_argument("--type", required=True, choices=["embedding", "explanation", "label", "report"], help="Job type")
    bf_create.add_argument("--total", type=int, default=1000, help="Total items to process")
    bf_create.set_defaults(func=cmd_backfill_create)

    bf_sub.add_parser("list", help="List all backfill jobs").set_defaults(func=cmd_backfill_list)

    bf_status = bf_sub.add_parser("status", help="Show job status")
    bf_status.add_argument("job_id", help="Job ID")
    bf_status.set_defaults(func=cmd_backfill_status)

    bf_pause = bf_sub.add_parser("pause", help="Pause a running job")
    bf_pause.add_argument("job_id", help="Job ID")
    bf_pause.set_defaults(func=cmd_backfill_pause)

    bf_resume = bf_sub.add_parser("resume", help="Resume a paused job")
    bf_resume.add_argument("job_id", help="Job ID")
    bf_resume.set_defaults(func=cmd_backfill_resume)


# ── Command implementations ──


def cmd_generate(args) -> None:
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print_error("Prompt is required (provide as argument or pipe)")
        return

    provider, cfg = get_provider(args.provider, args.model)
    kwargs = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    try:
        resp = provider.generate_detailed(prompt, **kwargs)
        if args.json:
            print_json({
                "text": resp.text,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
                "cost": resp.cost,
                "latency": resp.latency,
                "model": resp.model,
            })
        else:
            print_llm_response(resp.text, resp.total_tokens, resp.cost, resp.latency, resp.model)
    except Exception as e:
        print_error(str(e))


def cmd_chat(args) -> None:
    provider, cfg = get_provider(args.provider, args.model)
    kwargs = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    run_chat(provider, model=args.model or "", system_prompt=args.system_prompt, **kwargs)


def cmd_rag_query(args) -> None:
    provider, cfg = get_provider(args.provider, args.model)
    try:
        from astroml.llm.rag import RAGPipeline
        from astroml.llm.rag.retriever import Retriever
        from astroml.llm.embeddings import EmbeddingsService
        emb = EmbeddingsService()
        retriever = Retriever(embeddings_service=emb)
        pipeline = RAGPipeline(retriever=retriever, llm_provider=provider)
        answer, docs, meta = pipeline.query(args.question)
        if args.json:
            print_json({"answer": answer, "documents": [d.__dict__ for d in docs], "meta": meta})
        else:
            print_text(answer)
            if docs:
                print_text("\nSources:", format="markdown")
                for d in docs:
                    print_text(f"- {d.title} (score: {d.score:.3f})")
    except Exception as e:
        print_error(f"RAG query failed: {e}")


def cmd_embed(args) -> None:
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print_error("Text is required")
        return

    provider, _ = get_provider(args.provider or "openai")
    try:
        vector = provider.embed(text)
        if args.json:
            print_json({"text": text, "vector": vector, "dimensions": len(vector)})
        else:
            output_result(f"Embedding ({len(vector)} dimensions): {str(vector[:5])}...", as_json=False)
    except Exception as e:
        print_error(str(e))


def cmd_prompts_list(args) -> None:
    try:
        from astroml.llm.prompts import PromptRegistry
        registry = PromptRegistry()
        templates = registry.list_templates()
        if not templates:
            print_text("No prompt templates found.")
            return
        rows = [[name, ver or ""] for name, ver in templates.items()]
        print_table(rows, ["Name", "Version"], title="Prompt Templates")
    except Exception as e:
        print_error(str(e))


def cmd_prompts_render(args) -> None:
    try:
        from astroml.llm.prompts import PromptRegistry
        variables = {}
        for v in args.var:
            if "=" in v:
                k, val = v.split("=", 1)
                variables[k] = val
        registry = PromptRegistry()
        rendered = registry.render(args.name, variables)
        output_result(rendered)
    except Exception as e:
        print_error(str(e))


def cmd_prompts_test(args) -> None:
    provider, _ = get_provider(args.provider, args.model)
    try:
        from astroml.llm.prompts import PromptRegistry
        variables = {}
        for v in args.var:
            if "=" in v:
                k, val = v.split("=", 1)
                variables[k] = val
        registry = PromptRegistry()
        rendered = registry.render(args.name, variables)
        print_text(f"[dim]Rendered prompt:[/dim]\n{rendered}\n")
        resp = provider.generate_detailed(rendered)
        print_llm_response(resp.text, resp.total_tokens, resp.cost, resp.latency, resp.model)
    except Exception as e:
        print_error(str(e))


def cmd_eval_run(args) -> None:
    provider, _ = get_provider(args.provider, args.model)
    try:
        from astroml.llm.eval.benchmarks import BenchmarkRunner
        runner = BenchmarkRunner()
        results = runner.run(args.benchmark, provider)
        print_json(results)
    except Exception as e:
        print_error(str(e))


def cmd_eval_results(args) -> None:
    try:
        from astroml.llm.eval.framework import LLMEvalFramework
        framework = LLMEvalFramework(model_name="", generation_fn=None)
        history = framework.load_results()
        if not history:
            print_text("No evaluation results found.")
            return
        rows = [[r.get("benchmark", ""), str(r.get("score", "")), r.get("date", "")] for r in history]
        print_table(rows, ["Benchmark", "Score", "Date"], title="Evaluation History")
    except Exception as e:
        print_error(str(e))


def cmd_models(args) -> None:
    known_models = [
        ["openai", "gpt-4", "0.03 / 0.06"],
        ["openai", "gpt-4o", "0.01 / 0.03"],
        ["openai", "gpt-3.5-turbo", "0.0015 / 0.002"],
        ["anthropic", "claude-3-opus", "0.015 / 0.075"],
        ["anthropic", "claude-3-sonnet", "0.003 / 0.015"],
        ["anthropic", "claude-3-haiku", "0.00025 / 0.00125"],
        ["huggingface", "meta-llama/Llama-2-7b-chat-hf", "free (HF)"],
        ["local", "any local model", "free"],
    ]
    print_table(known_models, ["Provider", "Model", "Cost (input/output per 1K)"], title="Available Models")


def cmd_cost(args) -> None:
    try:
        from astroml.llm.cost.analytics import get_cost_summary
        summary = get_cost_summary()
        print_json(summary)
    except Exception as e:
        print_error(f"Cost tracking not available: {e}")


def cmd_cache(args) -> None:
    try:
        from astroml.llm.cache import CacheManager
        cache = CacheManager()
        stats = cache.get_stats()
        print_json(stats)
    except Exception as e:
        print_error(f"Cache not available: {e}")


# ── Backfill commands ──


def cmd_backfill_create(args) -> None:
    from astroml.llm.batch import get_scheduler
    scheduler = get_scheduler()
    job = scheduler.create_job(
        job_type=args.type,
        total_items=args.total,
        config={"type": args.type},
    )
    print_json(job)


def cmd_backfill_list(args) -> None:
    from astroml.llm.batch import get_scheduler
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()
    if not jobs:
        print_text("No backfill jobs found.")
        return
    rows = [[j["id"], j["job_type"], j["status"], str(j["total_items"]), str(j["processed_items"])] for j in jobs]
    print_table(rows, ["ID", "Type", "Status", "Total", "Processed"], title="Backfill Jobs")


def cmd_backfill_status(args) -> None:
    from astroml.llm.batch import get_scheduler
    scheduler = get_scheduler()
    job = scheduler.get_job(args.job_id)
    if job is None:
        print_error(f"Job '{args.job_id}' not found")
        return
    print_json(job)


def cmd_backfill_pause(args) -> None:
    from astroml.llm.batch import get_scheduler
    scheduler = get_scheduler()
    job = scheduler.pause_job(args.job_id)
    if job is None:
        print_error(f"Job '{args.job_id}' not found")
        return
    print_json(job)


def cmd_backfill_resume(args) -> None:
    from astroml.llm.batch import get_scheduler
    scheduler = get_scheduler()
    job = scheduler.resume_job(args.job_id)
    if job is None:
        print_error(f"Job '{args.job_id}' not found")
        return
    print_json(job)
