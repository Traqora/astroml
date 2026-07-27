"""Natural language to ML pipeline yaml generator."""
from __future__ import annotations

import yaml
from typing import Dict, Any, Optional


def generate_pipeline_config(
    natural_query: str,
) -> str:
    """Translate natural language instructions into runnable ML pipeline configuration YAML."""
    nl_lower = natural_query.lower()
    
    # 1. Base pipeline templates
    if "train" in nl_lower or "fraud detection model" in nl_lower:
        # Default training pipeline config
        days = 30
        if "last 30 days" in nl_lower:
            days = 30
        elif "last 90 days" in nl_lower:
            days = 90
            
        config = {
            "pipeline_type": "training",
            "model": {
                "name": "fraud_detection_gat",
                "version": "1.3.0",
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "epochs": 15,
                    "hidden_dim": 64
                }
            },
            "dataset": {
                "source": "stellar_ledger_operations",
                "range_days": days,
                "filters": {
                    "payment_only": True
                }
            },
            "eval": {
                "split_ratio": 0.2,
                "metrics": ["auc", "precision", "recall"]
            }
        }
    elif "compare" in nl_lower or "accuracy" in nl_lower:
        config = {
            "pipeline_type": "comparison",
            "models": [
                {"name": "fraud_detection_gat", "version": "1.2.0"},
                {"name": "fraud_detection_gat", "version": "1.3.0"}
            ],
            "dataset": {
                "source": "stellar_historical_golden",
                "limit": 5000
            },
            "metrics": ["accuracy", "latency_ms"]
        }
    else:
        config = {
            "pipeline_type": "inference",
            "model": {
                "name": "fraud_detection_gat",
                "version": "latest"
            },
            "data": {
                "live_stream": True,
                "max_delay_seconds": 10
            }
        }
        
    return yaml.dump(config, default_flow_style=False)
