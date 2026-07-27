"""Experiment result reporting and visualization."""

from typing import Dict, Any, List
from datetime import datetime


class ExperimentReporter:
    """
    Generate reports on experiment results.
    
    Produces summaries for stakeholders and automated decision-making.
    """

    def __init__(self, experiment_id: str, experiment_name: str):
        """Initialize reporter."""
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name

    def generate_summary_report(
        self,
        test_results: Dict[str, Any],
        statistical_results: Dict[str, Any],
    ) -> str:
        """
        Generate executive summary report.

        Args:
            test_results: A/B test results
            statistical_results: Statistical analysis results

        Returns:
            Formatted report as string
        """
        report = f"""
EXPERIMENT REPORT: {self.experiment_name}
{'='*60}

Experiment ID: {self.experiment_id}
Date: {datetime.now().isoformat()}

RESULTS SUMMARY
-----------------
Control Success Rate: {test_results['control_success_rate']:.1%}
Treatment Success Rate: {test_results['treatment_success_rate']:.1%}
Uplift: {(test_results['treatment_success_rate'] - test_results['control_success_rate']):.1%}

STATISTICAL SIGNIFICANCE
--------------------------
P-value: {statistical_results['p_value']:.4f}
Is Significant: {statistical_results['is_significant']}
Confidence Level: {(1 - statistical_results['p_value']):.1%}

RECOMMENDATION
---------------
Winner: {statistical_results.get('winner', 'No clear winner')}
Action: {'Deploy winner' if statistical_results['is_significant'] else 'Continue testing'}

{'='*60}
        """
        return report.strip()

    def generate_detailed_report(
        self,
        test_results: Dict[str, Any],
        statistical_results: Dict[str, Any],
        safety_metrics: Dict[str, Any] = None,
    ) -> str:
        """
        Generate detailed technical report.

        Args:
            test_results: A/B test results
            statistical_results: Statistical analysis
            safety_metrics: Safety metrics if available

        Returns:
            Detailed formatted report
        """
        report = self.generate_summary_report(test_results, statistical_results)

        if safety_metrics:
            report += f"""

SAFETY METRICS
---------------
Hallucination Rate: {safety_metrics.get('hallucination_rate', 'N/A')}
Toxicity Score: {safety_metrics.get('toxicity_score', 'N/A')}
Anomalies Detected: {safety_metrics.get('anomalies_detected', 'N/A')}
            """

        return report

    def export_results_csv(
        self,
        observations: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Export raw observations to CSV.

        Args:
            observations: List of observation records
            output_path: Path to save CSV
        """
        # Simulate CSV export
        csv_content = "user_id,variant,success,timestamp,metrics\n"
        for obs in observations[:10]:  # First 10 rows
            csv_content += f"{obs.get('user_id')},{obs.get('variant')},{obs.get('success')},{obs.get('timestamp')},\n"

        with open(output_path, 'w') as f:
            f.write(csv_content)

    def generate_metrics_table(
        self,
        metrics_by_variant: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Generate formatted metrics table.

        Args:
            metrics_by_variant: Dict mapping variant to metrics

        Returns:
            Formatted table as string
        """
        table = "Metric | "
        table += " | ".join(metrics_by_variant.keys())
        table += "\n"
        table += "------|" + "|".join(["---"] * len(metrics_by_variant)) + "\n"

        # Example metrics
        for metric in ["Success Rate", "Latency (ms)", "Cost"]:
            table += f"{metric} | "
            for variant in metrics_by_variant.values():
                table += f"{variant.get(metric, 'N/A')} | "
            table += "\n"

        return table
