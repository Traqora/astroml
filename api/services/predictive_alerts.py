"""Predictive alerts service for account behavior changes (Issue 2)."""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats

from api.database import get_sync_db
from api.models.orm import ApiTransaction
from astroml.llm.provider import MockLLMProvider

logger = logging.getLogger(__name__)

# Initialize LLM provider for generating explanations
llm_provider = MockLLMProvider()


class BehavioralLearner:
    """Learn behavioral baselines for account metrics."""

    def __init__(self, min_samples: int = 5, confidence_level: float = 0.95):
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        self._baselines: Dict[str, Dict[str, Dict[str, Any]]] = {}  # account_id -> metric -> stats

    def update_behavior(self, account_id: str, metrics: Dict[str, List[float]]) -> None:
        """Update behavioral baselines for an account based on historical data."""
        if account_id not in self._baselines:
            self._baselines[account_id] = {}

        for metric_name, values in metrics.items():
            if len(values) < self.min_samples:
                continue

            try:
                mean_val = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0.0
                min_val = min(values)
                max_val = max(values)

                # Calculate confidence interval
                if len(values) >= 2 and stdev > 0:
                    conf_interval = stats.t.interval(
                        self.confidence_level,
                        len(values) - 1,
                        loc=mean_val,
                        scale=stats.sem(values)
                    )
                else:
                    conf_interval = (mean_val, mean_val)

                self._baselines[account_id][metric_name] = {
                    "mean": mean_val,
                    "std_dev": stdev,
                    "min": min_val,
                    "max": max_val,
                    "sample_size": len(values),
                    "last_updated": datetime.utcnow(),
                    "confidence_interval": [float(ci) for ci in conf_interval],
                    "confidence_level": self.confidence_level
                }
            except Exception as e:
                logger.warning(f"Failed to calculate baseline for {account_id}.{metric_name}: {e}")

    def get_baseline(self, account_id: str, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get baseline for a specific account and metric."""
        return self._baselines.get(account_id, {}).get(metric_name)

    def get_all_baselines(self, account_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all baselines for an account."""
        return self._baselines.get(account_id, {})


class DeviationDetector:
    """Detect significant deviations from behavioral baselines."""

    def __init__(self, sensitivity: str = "medium"):
        self.sensitivity = sensitivity
        self.thresholds = {
            "low": 2.5,      # 2.5 sigma
            "medium": 2.0,   # 2.0 sigma
            "high": 1.5,     # 1.5 sigma
        }
        self.threshold = self.thresholds.get(sensitivity, 2.0)

    def detect_deviation(
        self,
        account_id: str,
        metric_name: str,
        current_value: float,
        baseline: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect if current value deviates significantly from baseline."""
        if not baseline:
            return None

        mean_val = baseline["mean"]
        stdev = baseline["std_dev"]

        if stdev == 0:
            # No variation in historical data
            if current_value != mean_val:
                deviation_score = float('inf') if current_value != mean_val else 0.0
            else:
                return None
        else:
            # Calculate z-score (absolute deviation)
            deviation_score = abs(current_value - mean_val) / stdev

        # Check if deviation exceeds threshold
        if deviation_score > self.threshold:
            # Get expected range (confidence interval)
            ci_low, ci_high = baseline.get("confidence_interval", [mean_val, mean_val])

            # Determine severity based on deviation magnitude
            if deviation_score >= 3.5:
                severity = "critical"
            elif deviation_score >= 2.5:
                severity = "high"
            elif deviation_score >= 1.5:
                severity = "medium"
            else:
                severity = "low"

            return {
                "alert_id": str(uuid4()),
                "account_id": account_id,
                "metric_name": metric_name,
                "current_value": current_value,
                "expected_range": [float(ci_low), float(ci_high)],
                "deviation_score": float(deviation_score),
                "severity": severity,
                "confidence": min(0.95, 0.5 + (deviation_score - self.threshold) * 0.1)
            }

        return None

    def detect_multiple_deviations(
        self,
        account_id: str,
        current_metrics: Dict[str, float],
        baselines: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect deviations across multiple metrics."""
        deviations = []

        for metric_name, current_value in current_metrics.items():
            baseline = baselines.get(metric_name)
            if baseline:
                deviation = self.detect_deviation(
                    account_id, metric_name, current_value, baseline
                )
                if deviation:
                    deviations.append(deviation)

        # Sort by deviation score (descending)
        deviations.sort(key=lambda x: x["deviation_score"], reverse=True)
        return deviations


class AlertGenerator:
    """Generate alerts and explanations using LLM."""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider or MockLLMProvider()
        self._explanation_cache: Dict[str, str] = {}

    def generate_explanation(
        self,
        alert_id: str,
        account_id: str,
        deviation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate natural language explanation for a deviation."""
        # Check cache first
        cache_key = f"{alert_id}:{hash(str(deviation_data))}"
        if cache_key in self._explanation_cache:
            return {"explanation": self._explanation_cache[cache_key]}

        try:
            # Build prompt for LLM
            prompt = f"""
            Explain this financial anomaly in clear, concise language suitable for a fraud analyst:
            
            Account: {account_id}
            Metric: {deviation_data.get('metric_name', 'unknown')}
            Current Value: {deviation_data.get('current_value', 0):.2f}
            Expected Range: [{deviation_data.get('expected_range', [0, 0])[0]:.2f}, {deviation_data.get('expected_range', [0, 0])[1]:.2f}]
            Deviation Score: {deviation_data.get('deviation_score', 0):.2f}
            Severity: {deviation_data.get('severity', 'unknown')}
            
            Provide a brief explanation of what this anomaly might indicate about account behavior.
            Keep it under 2 sentences and focus on the business implications.
            """

            # Generate explanation using LLM
            explanation_text = self.llm.generate(prompt)
            
            # Clean up the explanation
            explanation_text = explanation_text.strip()
            if not explanation_text.endswith('.'):
                explanation_text += '.'
                
            # Cache the result
            self._explanation_cache[cache_key] = explanation_text
            
            return {"explanation": explanation_text}
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for alert {alert_id}: {e}")
            return {
                "explanation": f"Anomaly detected in {deviation_data.get('metric_name', 'metric')} "
                              f"with deviation score of {deviation_data.get('deviation_score', 0):.2f}. "
                              f"This may indicate unusual account activity requiring investigation."
            }

    def create_deviation_alerts(
        self,
        deviations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert raw detections to formatted alert objects with explanations."""
        alerts = []
        for deviation in deviations:
            # Generate explanation
            explanation_result = self.generate_explanation(
                deviation["alert_id"],
                deviation["account_id"],
                deviation
            )
            
            # Create complete alert
            alert = deviation.copy()
            alert["explanation"] = explanation_result["explanation"]
            alerts.append(alert)
            
        return alerts


class PredictiveAlertService:
    """Main service for predictive alerts."""

    def __init__(self):
        self.behavioral_learner = BehavioralLearner()
        self.deviation_detector = DeviationDetector()
        self.alert_generator = AlertGenerator()
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes

    async def learn_behavior_from_transactions(
        self,
        account_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Learn behavioral baselines from historical transaction data."""
        try:
            db = next(get_sync_db())
            
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Query transactions for the account
            transactions = db.query(ApiTransaction).filter(
                ApiTransaction.source_account == account_id,
                ApiTransaction.created_at >= cutoff_date
            ).all()

            if not transactions:
                return {"message": "No transaction data found for learning period"}

            # Extract time-series metrics
            daily_metrics = defaultdict(list)
            
            for tx in transactions:
                date_key = tx.created_at.date()
                daily_metrics[date_key].append({
                    "amount": float(tx.amount or 0),
                    "hash": tx.hash
                })

            # Calculate daily aggregates
            daily_aggregates = {
                "daily_transaction_count": [],
                "daily_total_amount": [],
                "daily_avg_amount": [],
                "unique_counterparties_per_day": []
            }

            for date, txs in daily_metrics.items():
                amounts = [tx["amount"] for tx in txs]
                counterparts = list(set(tx.destination_account for tx in txs if tx.destination_account))
                
                daily_aggregates["daily_transaction_count"].append(len(txs))
                daily_aggregates["daily_total_amount"].append(sum(amounts))
                daily_averages = statistics.mean(amounts) if amounts else 0
                daily_aggregates["daily_avg_amount"].append(daily_averages)
                daily_aggregates["unique_counterparties_per_day"].append(len(counterparts))

            # Update behavioral models
            self.behavioral_learner.update_behavior(account_id, daily_aggregates)
            
            return {
                "account_id": account_id,
                "learning_period_days": days,
                "transactions_analyzed": len(transactions),
                "days_with_data": len(daily_metrics),
                "metrics_learned": list(daily_aggregates.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error learning behavior for account {account_id}: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    async def generate_predictive_alerts(
        self,
        account_id: str,
        lookback_days: int = 30,
        metrics: Optional[List[str]] = None,
        sensitivity: str = "medium"
    ) -> Dict[str, Any]:
        """Generate predictive alerts for an account."""
        try:
            # Update detector sensitivity
            self.deviation_detector.sensitivity = sensitivity
            self.deviation_detector.threshold = self.deviation_detector.thresholds[sensitivity]

            # Learn/update behavioral baselines
            learn_result = await self.learn_behavior_from_transactions(
                account_id, lookback_days
            )
            
            if "error" in result:
                return result

            # Get current metrics (last 24 hours)
            current_metrics = await self._get_current_metrics(account_id)
            
            if not current_metrics:
                return {
                    "message": "Insufficient recent data for analysis",
                    "account_id": account_id
                }

            # Get learned baselines
            baselines = self.behavioral_learner.get_all_baselines(account_id)
            
            if not baselines:
                return {
                    "message": "Insufficient historical data to establish baselines",
                    "account_id": account_id
                }

            # Filter metrics if specified
            if metrics:
                current_metrics = {k: v for k, v in current_metrics.items() if k in metrics}
                baselines = {k: v for k, v in baselines.items() if k in metrics}

            # Detect deviations
            deviations = self.deviation_detector.detect_multiple_deviations(
                account_id, current_metrics, baselines
            )

            # Create formatted alerts with explanations
            alerts = self.alert_generator.create_deviation_alerts(deviations)

            return {
                "alerts": alerts,
                "baselines_used": [
                    {
                        "account_id": account_id,
                        "metric_name": name,
                        "mean_value": data["mean"],
                        "std_dev": data["std_dev"],
                        "min_value": data["min"],
                        "max_value": data["max"],
                        "sample_size": data["sample_size"],
                        "last_updated": data["last_updated"].isoformat()
                    }
                    for name, data in baselines.items()
                ],
                "generated_at": datetime.utcnow().isoformat(),
                "total_analyzed": len(current_metrics),
                "deviations_found": len(deviations),
                "learning_info": learn_result
            }

        except Exception as e:
            logger.error(f"Error generating predictive alerts for {account_id}: {e}")
            return {"error": str(e)}

    async def _get_current_metrics(self, account_id: str) -> Dict[str, float]:
        """Get current metrics from recent transactions (last 24 hours)."""
        try:
            db = next(get_sync_db())
            
            # Get transactions from last 24 hours
            cutoff_date = datetime.utcnow() - timedelta(hours=24)
            
            transactions = db.query(ApiTransaction).filter(
                ApiTransaction.source_account == account_id,
                ApiTransaction.created_at >= cutoff_date
            ).all()

            if not transactions:
                return {}

            # Calculate current metrics
            amounts = [float(tx.amount or 0) for tx in transactions]
            counterparties = list(set(tx.destination_account for tx in transactions if tx.destination_account))
            
            return {
                "transaction_count": len(transactions),
                "total_amount": sum(amounts),
                "avg_amount": statistics.mean(amounts) if amounts else 0,
                "max_amount": max(amounts) if amounts else 0,
                "unique_counterparties": len(counterparties)
            }

        except Exception as e:
            logger.error(f"Error getting current metrics for {account_id}: {e}")
            return {}
        finally:
            db.close()

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of the predictive alerts service."""
        total_accounts = len(self.behavioral_learner._baselines)
        total_models = sum(len(metrics) for metrics in self.behavioral_learner._baselines.values())
        
        return {
            "service": "predictive_alerts",
            "status": "active",
            "models_learned": {
                "accounts": total_accounts,
                "total_baselines": total_models
            },
            "cache_size": len(self._cache),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
predictive_alert_service = PredictiveAlertService()