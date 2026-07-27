"""Explanation prompt templates."""
from typing import Any, Dict, List


class ExplanationTemplates:
    """Templates for generating explanation prompts."""

    FRAUD_ALERT_TEMPLATE = """Explain why this Stellar account was flagged for potential fraud.

Account ID: {account_id}
Risk Score: {risk_score:.2f} / 1.0
Risk Level: {risk_level}
Pattern Detected: {pattern}

Recent Transaction Evidence:
{transactions}

Provide a {detail_level} explanation that includes:
1. Why this account was flagged
2. What suspicious patterns were detected
3. The severity of the risk
4. Recommended actions

Keep the explanation factual and cite specific transactions as evidence."""

    MODEL_PREDICTION_TEMPLATE = """Explain this machine learning model prediction.

Model: {model_name}
Prediction: {prediction}
Confidence: {confidence:.1f}%

Top Contributing Features:
{features}

Provide a {detail_level} explanation that includes:
1. What the model predicted and why
2. Which features were most important
3. How confident the model is
4. What would change the prediction

Target audience: {audience}"""

    ANOMALY_DETECTION_TEMPLATE = """Explain this transaction anomaly detection result.

Transaction ID: {transaction_id}
From: {source_account}
To: {destination_account}
Amount: {amount} {asset_code}
Anomaly Score: {anomaly_score:.2f}
Anomaly Type: {anomaly_type}

Historical Context:
{context}

Graph Patterns:
{graph_patterns}

Provide a {detail_level} explanation that includes:
1. What makes this transaction anomalous
2. How it compares to historical patterns
3. Relevant graph relationships
4. Potential explanations for the anomaly

Keep it factual and evidence-based."""

    FEATURE_ATTRIBUTION_TEMPLATE = """Explain the feature importance for this prediction.

Features and Contributions:
{feature_details}

Provide insights on:
1. Which features had the strongest impact
2. Whether they pushed toward or away from the prediction
3. How these features interact
4. Any surprising patterns"""

    EXECUTIVE_SUMMARY_TEMPLATE = """Summarize this {explanation_type} in 2-3 sentences suitable for non-technical stakeholders.

Full Explanation:
{full_explanation}

Create an executive summary that:
- Uses simple language
- Focuses on business impact
- Includes actionable insights"""

    @staticmethod
    def format_transactions(transactions: List[Dict[str, Any]], max_count: int = 5) -> str:
        """Format transaction list for template."""
        lines = []
        for i, tx in enumerate(transactions[:max_count], 1):
            line = f"{i}. Tx {tx.get('hash', 'N/A')[:12]}...: {tx.get('amount', 0)} {tx.get('asset_code', 'XLM')}"
            line += f" to {tx.get('destination_account', 'N/A')[:12]}..."
            line += f" (Ledger: {tx.get('ledger_sequence', 'N/A')})"
            lines.append(line)
        return "\n".join(lines) if lines else "No transactions available"

    @staticmethod
    def format_features(features: Dict[str, float], max_count: int = 10) -> str:
        """Format feature contributions for template."""
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        lines = []
        for i, (name, value) in enumerate(sorted_features[:max_count], 1):
            direction = "+" if value > 0 else ""
            lines.append(f"{i}. {name}: {direction}{value:.3f}")
        return "\n".join(lines) if lines else "No features available"

    @staticmethod
    def format_graph_patterns(patterns: List[str]) -> str:
        """Format graph patterns for template."""
        if not patterns:
            return "No specific graph patterns identified"
        return "\n".join(f"- {pattern}" for pattern in patterns)
