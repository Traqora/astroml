import re

class QueryTranslator:
    def __init__(self):
        self.sql_injection_patterns = [
            r"(?i)\bDROP\b", r"(?i)\bDELETE\b", r"(?i)\bUPDATE\b",
            r"(?i)\bINSERT\b", r"(?i)\bALTER\b", r"(?i)\bTRUNCATE\b",
            r"(?i)\bEXEC\b", r"(?i)\bUNION\b", r"--", r";"
        ]

    def _is_safe(self, query: str) -> bool:
        """Check for basic SQL injection attempts."""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, query):
                return False
        return True

    def translate_to_sql(self, nl_query: str) -> str:
        """
        Translate natural language to SQL.
        In a real scenario, this uses an LLM. Here, we mock it.
        """
        if not self._is_safe(nl_query):
            raise ValueError("Potential SQL injection detected.")
            
        # Intent recognition & entity extraction (mocked)
        nl_lower = nl_query.lower()
        if "fraud" in nl_lower:
            return "SELECT * FROM transactions WHERE status = 'fraud' LIMIT 10;"
        elif "amount greater than" in nl_lower:
            # Extract number
            match = re.search(r"amount greater than (\d+)", nl_lower)
            amount = match.group(1) if match else "1000"
            return f"SELECT * FROM transactions WHERE amount > {amount};"
            
        return "SELECT * FROM transactions LIMIT 50;"
