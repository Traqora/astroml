from fastapi import APIRouter
router = APIRouter()
@router.get("/synthetic")
def get_synthetic() -> dict:
    return {"status": "ok"}
