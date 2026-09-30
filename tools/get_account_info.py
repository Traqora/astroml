"""Tool: Fetch Stellar account details."""

from typing import Any
from astroml.llm.tools.definitions import BaseTool


class GetAccountInfoTool(BaseTool):
    name = "get_account_info"
    description = "Fetch detailed information about a Stellar blockchain account by its public key."
    parameters = {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "Stellar account public key (G...)",
            },
        },
        "required": ["account_id"],
    }

    async def execute(self, params: dict[str, Any]) -> Any:
        account_id = params["account_id"]
        if not account_id.startswith("G") or len(account_id) != 56:
            return {"error": "Invalid Stellar account ID"}
        return {
            "account_id": account_id,
            "balance": "100.0000000",
            "sequence": "123456789",
            "subentry_count": 2,
            "last_activity": "2026-07-25T00:00:00Z",
            "note": "Account info fetched from stub — connect to Horizon for live data",
        }
