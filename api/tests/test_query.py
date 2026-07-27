"""Integration tests for query optimization API endpoints."""
from __future__ import annotations

class TestQueryOptimization:
    def test_query_optimization_returns_200_and_valid_json(self, client):
        sql = "SELECT * FROM accounts WHERE id = '123' JOIN transactions ON transactions.account_id = accounts.id"
        resp = client.get("/api/v1/query/optimize", params={"query": sql})
        assert resp.status_code == 200
        
        data = resp.json()
        assert "original_query" in data
        assert "optimized_query" in data
        assert "suggested_indexes" in data
        assert "explanation" in data
        assert "estimated_time_saving" in data
        
        # Verify estimated time savings is > 30%
        assert data["estimated_time_saving"] > 30
        
        # Verify relevant indexes are suggested
        assert len(data["suggested_indexes"]) >= 1
        assert any("CREATE INDEX" in idx for idx in data["suggested_indexes"])

    def test_empty_query_returns_400(self, client):
        resp = client.get("/api/v1/query/optimize", params={"query": "   "})
        assert resp.status_code == 400
