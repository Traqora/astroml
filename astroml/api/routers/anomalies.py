from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from astroml.llm.anomaly_explanation import AnomalyExplanationEngine

router = APIRouter(prefix="/api/v1/anomalies", tags=["Anomalies"])

# Dependency injection for the explanation engine
def get_explanation_engine():
    # In a real scenario, you'd pass the actual LLM provider here
    return AnomalyExplanationEngine(llm_provider=None)

@router.get("/{anomaly_id}/explanation")
async def get_anomaly_explanation(
    anomaly_id: str,
    account_id: str,
    engine: AnomalyExplanationEngine = Depends(get_explanation_engine)
) -> Dict[str, Any]:
    """
    Generate an explanation for a detected anomaly.
    """
    try:
        # Mock fetching anomaly data from DB
        anomaly_data = {
            "tx_volume": 15000,
            "unique_counterparties": 45,
            "velocity": 5.8,
            "time_of_day": "03:00 AM"
        }
        
        explanation = engine.generate_explanation(anomaly_id, account_id, anomaly_data)
        return {
            "status": "success",
            "data": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
