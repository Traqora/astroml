"""LLM-based report generator (issue XXX)."""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.orm import ApiTransaction, FraudAlert, ApiAccount
from api.services.llm_context import MultiModalContextHandler

context_handler = MultiModalContextHandler()


@dataclass
class ReportData:
    start_date: datetime
    end_date: datetime
    transactions: List[Dict[str, Any]]
    fraud_alerts: List[Dict[str, Any]]
    accounts: List[Dict[str, Any]]


class ReportGenerator:
    """Generates Markdown and PDF reports with LLM insights and charts."""

    def __init__(self):
        pass

    async def fetch_report_data(self, db: AsyncSession, days: int = 90) -> ReportData:
        """Fetch data for the report from the database."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Fetch transactions
        tx_result = await db.execute(
            select(ApiTransaction)
            .where(ApiTransaction.created_at >= start_date)
            .order_by(ApiTransaction.created_at.desc())
            .limit(1000)
        )
        transactions = []
        for tx in tx_result.scalars().all():
            transactions.append({
                "hash": tx.hash,
                "source_account": tx.source_account,
                "destination_account": tx.destination_account,
                "amount": tx.amount,
                "asset_code": tx.asset_code,
                "created_at": tx.created_at.isoformat(),
            })

        # Fetch fraud alerts
        fraud_result = await db.execute(
            select(FraudAlert)
            .where(FraudAlert.detected_at >= start_date)
            .order_by(FraudAlert.detected_at.desc())
        )
        fraud_alerts = []
        for alert in fraud_result.scalars().all():
            fraud_alerts.append({
                "account_id": alert.account_id,
                "risk_level": alert.risk_level,
                "risk_score": alert.risk_score,
                "description": alert.description,
                "detected_at": alert.detected_at.isoformat(),
            })

        # Fetch active accounts
        accounts_result = await db.execute(
            select(ApiAccount)
            .where(ApiAccount.last_active >= start_date)
            .order_by(ApiAccount.last_active.desc())
            .limit(100)
        )
        accounts = []
        for acc in accounts_result.scalars().all():
            accounts.append({
                "public_key": acc.public_key,
                "balance": acc.balance,
                "last_active": acc.last_active.isoformat() if acc.last_active else None,
            })

        return ReportData(
            start_date=start_date,
            end_date=end_date,
            transactions=transactions,
            fraud_alerts=fraud_alerts,
            accounts=accounts,
        )

    def _generate_chart_base64(self, data: List[Dict[str, Any]], chart_type: str = "bar") -> str:
        """Generate a simple chart as base64 image (placeholder)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(10, 6))
            
            if chart_type == "bar" and data:
                labels = [f"{i+1}" for i in range(len(data))]
                values = [d.get("amount", d.get("risk_score", 1)) for d in data]
                plt.bar(labels, values[:20], color="skyblue")
                plt.title(f"{chart_type.capitalize()} Chart")
                plt.xlabel("Items")
                plt.ylabel("Value")
                plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()
            return base64.b64encode(buf.read()).decode("utf-8")
        except ImportError:
            return ""

    def generate_markdown(self, report_data: ReportData, llm_summary: Optional[str] = None) -> str:
        """Generate Markdown report."""
        # Generate charts
        tx_chart = self._generate_chart_base64(report_data.transactions, "bar")
        fraud_chart = self._generate_chart_base64(report_data.fraud_alerts, "bar")

        markdown = f"""# AstroML Report - {datetime.utcnow().strftime('%Y-%m-%d')}

## Executive Summary

{llm_summary or "LLM-generated summary will appear here. This is a placeholder summary with key insights from the 90-day transaction history, fraud detection, and account activity."}

---

## Transaction Overview

- **Time Period**: {report_data.start_date.strftime('%Y-%m-%d')} to {report_data.end_date.strftime('%Y-%m-%d')}
- **Total Transactions**: {len(report_data.transactions)}
- **Total Volume**: ${sum(tx.get('amount', 0) for tx in report_data.transactions if tx.get('amount')):.2f}

### Transaction Chart
![Transaction Chart](data:image/png;base64,{tx_chart})

---

## Fraud Detection

- **Total Alerts**: {len(report_data.fraud_alerts)}
- **High Risk**: {len([a for a in report_data.fraud_alerts if a['risk_level'] == 'high'])}
- **Medium Risk**: {len([a for a in report_data.fraud_alerts if a['risk_level'] == 'medium'])}
- **Low Risk**: {len([a for a in report_data.fraud_alerts if a['risk_level'] == 'low'])}

### Fraud Chart
![Fraud Chart](data:image/png;base64,{fraud_chart})

---

## Active Accounts

- **Active Accounts**: {len(report_data.accounts)}

---

## Key Transactions

| Hash | Source | Destination | Amount | Asset | Date |
|------|--------|-------------|--------|-------|------|
"""

        for tx in report_data.transactions[:10]:
            markdown += f"| {tx['hash'][:16]}... | {tx['source_account'][:8]}... | {tx['destination_account'][:8] if tx['destination_account'] else 'N/A'}... | ${tx.get('amount', 0):.2f} | {tx['asset_code'] or 'XLM'} | {tx['created_at'][:10]} |\n"

        markdown += """
---

*Generated by AstroML Report Generator*
"""
        return markdown

    async def generate_pdf(self, markdown_content: str) -> bytes:
        """Generate PDF from Markdown (placeholder)."""
        try:
            import markdown
            from weasyprint import HTML

            html_content = markdown.markdown(markdown_content, extensions=["tables"])
            html = HTML(string=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #1a73e8; }}
                        h2 {{ color: #202124; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                        th {{ background-color: #f8f9fa; }}
                        img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
            """)
            return html.write_pdf()
        except ImportError:
            return b""


report_generator = ReportGenerator()
