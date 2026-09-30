from fastapi import APIRouter

router = APIRouter()
@router.get("/explainability")
def get_explainability() -> dict:
    return {"status": "ok"}
