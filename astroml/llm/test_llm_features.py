"""Tests for the new LLM features (secrets, rate limits, budgets, fallbacks)."""
import unittest
from unittest.mock import MagicMock, patch
import os
import time

from astroml.llm.secrets import encrypt_key, decrypt_key, store_api_key, get_api_key, rotate_api_key
from astroml.llm.exceptions import RateLimitExceededError, CostBudgetExceededError, ConfigurationError
from astroml.llm.rate_limiter import ProviderRateLimiter, CostBudgetManager
from astroml.llm.providers.factory import get_llm_provider
from astroml.llm.providers.openai import OpenAIProvider
from astroml.llm.providers.anthropic import AnthropicProvider

class SecretsTests(unittest.TestCase):
    def test_encryption_decryption(self):
        plain = "sk-test-key-12345"
        encrypted = encrypt_key(plain)
        self.assertNotEqual(plain, encrypted)
        
        decrypted = decrypt_key(encrypted)
        self.assertEqual(plain, decrypted)

    def test_store_and_get_api_key(self):
        store_api_key("test_provider", "api-key-value")
        key = get_api_key("test_provider")
        self.assertEqual(key, "api-key-value")

    def test_rotate_api_key(self):
        store_api_key("test_provider", "first-key")
        self.assertEqual(get_api_key("test_provider"), "first-key")
        
        rotate_api_key("test_provider", "second-key")
        self.assertEqual(get_api_key("test_provider"), "second-key")


class RateLimiterTests(unittest.TestCase):
    def test_request_rate_limiting(self):
        limiter = ProviderRateLimiter(requests_per_minute=2, tokens_per_minute=0)
        
        # First 2 requests succeed
        limiter.check_and_record(10)
        limiter.check_and_record(10)
        
        # Third request exceeds requests/min limit
        with self.assertRaises(RateLimitExceededError):
            limiter.check_and_record(10)

    def test_token_rate_limiting(self):
        limiter = ProviderRateLimiter(requests_per_minute=0, tokens_per_minute=50)
        
        # First request consumes 30 tokens
        limiter.check_and_record(30)
        
        # Second request of 25 tokens exceeds the 50 token limit
        with self.assertRaises(RateLimitExceededError):
            limiter.check_and_record(25)


class CostBudgetTests(unittest.TestCase):
    def test_cost_budget_enforcement(self):
        manager = CostBudgetManager(daily_limit=1.0, monthly_limit=5.0)
        
        # Spends under budget
        manager.record_spend(0.50)
        manager.check_budget() # Should pass
        
        # Spends exactly daily limit
        manager.record_spend(0.50)
        with self.assertRaises(CostBudgetExceededError):
            manager.check_budget()

    def test_cost_budget_alerts(self):
        manager = CostBudgetManager(daily_limit=10.0, monthly_limit=100.0)
        
        with self.assertLogs('astroml.llm.rate_limiter', level='WARNING') as cm:
            manager.record_spend(8.5) # 85%, triggers 80% warning
            manager.record_spend(2.0) # total 10.5, triggers 100% warning
            
        self.assertTrue(any("80% daily cost budget reached" in log for log in cm.output))
        self.assertTrue(any("100% daily cost budget reached" in log for log in cm.output))


class FallbackChainTests(unittest.TestCase):
    @patch('astroml.llm.providers.openai.OpenAIProvider._generate_raw')
    @patch('astroml.llm.providers.anthropic.AnthropicProvider._generate_raw')
    def test_fallback_automatic_failover(self, mock_anthropic_gen, mock_openai_gen):
        # Configure primary (OpenAI) to raise a transient exception
        mock_openai_gen.side_effect = Exception("500 Internal Server Error")
        mock_anthropic_gen.return_value = "hello from fallback claude"

        # Initialize provider with OpenAI as primary
        provider = get_llm_provider("openai")
        
        # Attach Anthropic as a fallback provider manually for the test
        anthropic_prov = AnthropicProvider(api_key="mock-key")
        provider.fallback_providers = [anthropic_prov]

        # Call generate (which calls generate_detailed)
        response_text = provider.generate("hi")
        
        # Verify it fallback successfully and returned Claude's response
        self.assertEqual(response_text, "hello from fallback claude")
        self.assertEqual(mock_openai_gen.call_count, 4) # 1 initial try + 3 retries
        mock_anthropic_gen.assert_called_once()


class ModelInterpretabilityTests(unittest.TestCase):
    def test_shap_values(self):
        from astroml.llm.explainer import get_shap_values
        features = {"amount": 800.0, "velocity": 4.0, "unique_counterparties": 6}
        baseline = {"amount": 100.0, "velocity": 1.0, "unique_counterparties": 2}
        shap = get_shap_values(features, baseline)
        self.assertIn("amount", shap)
        self.assertGreater(shap["amount"], 0)
        self.assertGreater(shap["velocity"], 0)

    def test_decision_tree(self):
        from astroml.llm.explainer import generate_decision_tree
        features = {"amount": 800.0, "velocity": 4.0, "unique_counterparties": 6}
        tree = generate_decision_tree(features)
        self.assertEqual(tree["final_decision"], "Fraud Suspected")
        self.assertIn("digraph", tree["exportable_dot"])

    def test_attention_visualization(self):
        from astroml.llm.explainer import get_attention_visualization
        text = "This transaction is suspicious because velocity spiked and fraud is suspected."
        viz = get_attention_visualization(text)
        self.assertIn("html", viz)
        
        # Check tokens
        tokens = [t["token"] for t in viz["tokens"]]
        self.assertIn("suspicious", tokens)
        
        # The attention score for 'suspicious' is 0.90
        att_suspicious = next(t["attention"] for t in viz["tokens"] if t["token"] == "suspicious")
        self.assertEqual(att_suspicious, 0.90)

    def test_generate_explanation_report(self):
        from astroml.llm.explainer import generate_explanation_report
        features = {"amount": 800.0, "velocity": 4.0, "unique_counterparties": 6}
        baseline = {"amount": 100.0, "velocity": 1.0, "unique_counterparties": 2}
        report = generate_explanation_report(123, "acc-456", features, baseline, "Spiked velocity and fraud suspected.")
        self.assertIn("SHAP Values", report)
        self.assertIn("Decision Tree Path", report)
        self.assertIn("Attention Visualization Heatmap", report)


if __name__ == "__main__":
    unittest.main()
