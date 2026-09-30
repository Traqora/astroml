from fastapi import APIRouter

router = APIRouter()
@router.get("/forecast")
def get_forecast() -> dict:
    return {"status": "ok"}
