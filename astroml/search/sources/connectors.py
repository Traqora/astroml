import os
import glob
from typing import List, Dict, Any

class BaseSourceConnector:
    def __init__(self, name: str):
        self.name = name

    def fetch_documents(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

class DocsConnector(BaseSourceConnector):
    def __init__(self, docs_dir: str = "docs"):
        super().__init__("documentation")
        self.docs_dir = docs_dir

    def fetch_documents(self) -> List[Dict[str, Any]]:
        docs = []
        pattern = os.path.join(self.docs_dir, "**", "*.md")
        # Try to find real markdown files
        found_files = glob.glob(pattern, recursive=True)
        for path in found_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append({
                    "id": f"doc_{os.path.basename(path)}",
                    "title": os.path.basename(path),
                    "content": content,
                    "type": "documentation",
                    "path": path,
                    "metadata": {"author": "AstroML Team", "date": "2026-07-25"}
                })
            except Exception:
                continue
        # Fallback default docs if none found
        if not docs:
            docs.append({
                "id": "doc_intro",
                "title": "Introduction to AstroML Platform",
                "content": "AstroML is an advanced machine learning platform for real-time fraud detection, transaction analysis, and model monitoring.",
                "type": "documentation",
                "path": "docs/intro.md",
                "metadata": {"author": "Admin", "date": "2026-07-25"}
            })
        return docs

class TransactionsConnector(BaseSourceConnector):
    def __init__(self):
        super().__init__("transactions")

    def fetch_documents(self) -> List[Dict[str, Any]]:
        # Mock transaction records
        return [
            {
                "id": "tx_1001",
                "title": "Transaction 1001 - Large Withdrawal",
                "content": "Withdrawal of $50,000 from account ACT-4921 to unknown offshore account, flagged for high-risk pattern.",
                "type": "transaction",
                "metadata": {"amount": 50000.0, "account": "ACT-4921", "date": "2026-07-25"}
            },
            {
                "id": "tx_1002",
                "title": "Transaction 1002 - Coffee Shop",
                "content": "Routine POS purchase of $4.50 at local coffee shop, low risk account pattern.",
                "type": "transaction",
                "metadata": {"amount": 4.50, "account": "ACT-1102", "date": "2026-07-25"}
            }
        ]

class AlertsConnector(BaseSourceConnector):
    def __init__(self):
        super().__init__("alerts")

    def fetch_documents(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "alert_400",
                "title": "Alert 400 - Multiple Rapid Transits",
                "content": "Fraud alert triggered: card scanned in New York and Paris within 10 minutes (impossible travel speed).",
                "type": "alert",
                "metadata": {"severity": "critical", "source": "impossible_travel", "date": "2026-07-25"}
            },
            {
                "id": "alert_401",
                "title": "Alert 401 - Unusual Activity",
                "content": "Suspicious pattern: account with unusual activity, multiple micro-transfers to international routes.",
                "type": "alert",
                "metadata": {"severity": "warning", "source": "micro_transfers", "date": "2026-07-25"}
            }
        ]

class ModelsConnector(BaseSourceConnector):
    def __init__(self):
        super().__init__("models")

    def fetch_documents(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "model_xgb",
                "title": "XGBoost Fraud Classifier v2",
                "content": "Production fraud classification model utilizing gradient boosted trees. F1-Score: 0.91, AUC: 0.94. Handles 1000 requests/sec.",
                "type": "model",
                "metadata": {"framework": "xgboost", "accuracy": 0.95, "date": "2026-07-25"}
            }
        ]

class CodeConnector(BaseSourceConnector):
    def __init__(self):
        super().__init__("code")

    def fetch_documents(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "code_detector",
                "title": "fraud_detector.py",
                "content": "def detect_anomaly(transaction: dict) -> bool:\n    \"\"\"Run anomaly detection on incoming transactions using isolation forests.\"\"\"\n    pass",
                "type": "code",
                "metadata": {"language": "python", "lines": 42, "date": "2026-07-25"}
            }
        ]
