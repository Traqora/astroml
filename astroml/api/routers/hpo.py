from fastapi import APIRouter

router = APIRouter()
@router.get("/hpo")
def get_hpo() -> dict:
    return {"status": "ok"}
