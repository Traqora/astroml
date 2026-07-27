"""Tests for the LLM provider abstraction layer (issue #359)."""
import sys
import types
import unittest
from unittest.mock import MagicMock

from .factory import get_llm_provider, _PROVIDERS


def _install_fake_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class FactoryTests(unittest.TestCase):
    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            get_llm_provider("not-a-provider")

    def test_known_providers_registered(self):
        self.assertEqual(set(_PROVIDERS), {"openai", "anthropic", "huggingface", "local"})

    def test_switch_provider_via_config_only(self):
        import os

        os.environ["LLM_PROVIDER"] = "anthropic"
        try:
            provider = get_llm_provider(api_key="sk-test")
        finally:
            del os.environ["LLM_PROVIDER"]
        self.assertEqual(provider.__class__.__name__, "AnthropicProvider")


class SameInterfaceTests(unittest.TestCase):
    """Each provider must expose the same generate()/get_token_usage() interface."""

    def setUp(self):
        self._orig_modules = dict(sys.modules)

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._orig_modules)

    def test_openai_provider_generate(self):
        fake_choice = MagicMock()
        fake_choice.message.content = "hello from openai"
        fake_response = MagicMock(choices=[fake_choice])
        fake_response.usage.prompt_tokens = 5
        fake_response.usage.completion_tokens = 7
        fake_response.usage.total_tokens = 12
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        fake_openai = _install_fake_module("openai")
        fake_openai.OpenAI = MagicMock(return_value=fake_client)

        provider = get_llm_provider("openai", api_key="sk-test")
        text = provider.generate("hi")

        self.assertEqual(text, "hello from openai")
        self.assertEqual(provider.get_token_usage(), {
            "prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12,
        })

    def test_anthropic_provider_generate(self):
        fake_block = MagicMock(text="hello from anthropic")
        fake_response = MagicMock(content=[fake_block])
        fake_response.usage.input_tokens = 6
        fake_response.usage.output_tokens = 9
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        fake_anthropic = _install_fake_module("anthropic")
        fake_anthropic.Anthropic = MagicMock(return_value=fake_client)

        provider = get_llm_provider("anthropic", api_key="sk-test")
        text = provider.generate("hi")

        self.assertEqual(text, "hello from anthropic")
        self.assertEqual(provider.get_token_usage(), {
            "prompt_tokens": 6, "completion_tokens": 9, "total_tokens": 15,
        })

    def test_huggingface_provider_generate(self):
        fake_client = MagicMock()
        fake_client.text_generation.return_value = "hello from huggingface"
        fake_hf = _install_fake_module("huggingface_hub")
        fake_hf.InferenceClient = MagicMock(return_value=fake_client)

        provider = get_llm_provider("huggingface", api_key="hf-test")
        text = provider.generate("hi")

        self.assertEqual(text, "hello from huggingface")
        usage = provider.get_token_usage()
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])


if __name__ == "__main__":
    unittest.main()
