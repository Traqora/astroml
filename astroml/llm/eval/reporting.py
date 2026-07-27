"""Reporting utility to format and save LLM evaluation run results."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, List


def generate_eval_report(
    runner_model: str,
    results: List[Dict[str, Any]],
    output_dir: str = "benchmark_results"
) -> str:
    """Generate and write a markdown report visualizing evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_filename = f"eval_report_{runner_model.replace('-', '_')}_{timestamp}.md"
    report_path = os.path.join(output_dir, report_filename)
    
    # Calculate aggregate statistics
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    avg_latency = sum(r["latency_seconds"] for r in results) / total if total > 0 else 0.0
    
    # Aggregate metrics
    metrics_sum: Dict[str, float] = {}
    metrics_count = 0
    
    for r in results:
        if r["status"] != "success":
            continue
        metrics_count += 1
        for k, v in r.get("metrics", {}).items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            
    avg_metrics = {k: v / metrics_count for k, v in metrics_sum.items()} if metrics_count > 0 else {}
    
    report_content = f"""# LLM Evaluation Report for model: {runner_model}
Generated on: {datetime.utcnow().isoformat()}

## Summary Statistics
- **Total Test Cases**: {total}
- **Successful Runs**: {successful} / {total}
- **Average Latency**: {avg_latency:.4f}s

## Quality Performance Metrics
| Metric | Average Score |
| --- | --- |
"""
    for k, v in avg_metrics.items():
        report_content += f"| {k} | {v:.4f} |\n"
        
    report_content += "\n## Detailed Test Cases\n"
    for r in results:
        report_content += f"""
### Test Case {r['test_case_id']}
- **Prompt**: `{r['prompt']}`
- **Status**: {r['status']}
- **Latency**: {r['latency_seconds']:.4f}s
- **Response**: "{r['response']}"
- **Reference**: "{r['reference']}"
- **Metrics**: {json.dumps(r.get('metrics', {}))}
---
"""
        
    with open(report_path, "w") as f:
        f.write(report_content)
        
    return report_path
