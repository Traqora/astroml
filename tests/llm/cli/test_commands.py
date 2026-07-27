"""Tests for LLM CLI command registration."""

import argparse
from astroml.cli_llm.commands import register_llm_subcommands


class TestCliCommands:
    def setup_method(self):
        self.parser = argparse.ArgumentParser()
        self.sub = self.parser.add_subparsers(dest="llm_command", required=True)
        register_llm_subcommands(self.sub)

    def test_generate_subcommand_exists(self):
        args = self.parser.parse_args(["generate", "test prompt"])
        assert args.llm_command == "generate"
        assert hasattr(args, "func")

    def test_chat_subcommand_exists(self):
        args = self.parser.parse_args(["chat"])
        assert args.llm_command == "chat"
        assert hasattr(args, "func")

    def test_rag_query_subcommand_exists(self):
        args = self.parser.parse_args(["rag", "query", "test question"])
        assert args.rag_command == "query"
        assert hasattr(args, "func")

    def test_embed_subcommand_exists(self):
        args = self.parser.parse_args(["embed", "hello world"])
        assert args.llm_command == "embed"
        assert hasattr(args, "func")

    def test_prompts_list_subcommand_exists(self):
        args = self.parser.parse_args(["prompts", "list"])
        assert args.prompts_command == "list"
        assert hasattr(args, "func")

    def test_prompts_render_subcommand_exists(self):
        args = self.parser.parse_args(["prompts", "render", "test_template", "--var", "key=val"])
        assert args.prompts_command == "render"
        assert args.name == "test_template"
        assert args.var == ["key=val"]

    def test_prompts_test_subcommand_exists(self):
        args = self.parser.parse_args(["prompts", "test", "test_template"])
        assert args.prompts_command == "test"
        assert args.name == "test_template"

    def test_models_subcommand_exists(self):
        args = self.parser.parse_args(["models"])
        assert args.llm_command == "models"
        assert hasattr(args, "func")

    def test_eval_run_subcommand_exists(self):
        args = self.parser.parse_args(["eval", "run", "test_benchmark"])
        assert args.eval_command == "run"
        assert args.benchmark == "test_benchmark"

    def test_eval_results_subcommand_exists(self):
        args = self.parser.parse_args(["eval", "results"])
        assert args.eval_command == "results"
        assert hasattr(args, "func")

    def test_cost_subcommand_exists(self):
        args = self.parser.parse_args(["cost"])
        assert args.llm_command == "cost"
        assert hasattr(args, "func")

    def test_cache_subcommand_exists(self):
        args = self.parser.parse_args(["cache"])
        assert args.llm_command == "cache"
        assert hasattr(args, "func")

    def test_generate_with_flags(self):
        args = self.parser.parse_args([
            "generate", "test", "--provider", "anthropic", "--model", "claude-3",
            "--max-tokens", "500", "--temperature", "0.5", "--json",
        ])
        assert args.provider == "anthropic"
        assert args.model == "claude-3"
        assert args.max_tokens == 500
        assert args.temperature == 0.5
        assert args.json is True

    def test_chat_with_flags(self):
        args = self.parser.parse_args([
            "chat", "--provider", "anthropic", "--model", "claude-3",
            "--system-prompt", "Be helpful",
        ])
        assert args.provider == "anthropic"
        assert args.model == "claude-3"
        assert args.system_prompt == "Be helpful"
