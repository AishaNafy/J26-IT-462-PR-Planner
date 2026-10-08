from fastapi import APIRouter

router = APIRouter(prefix="/disease-detection", tags=["disease-detection"])


@router.get("/")
def list_disease_cases() -> dict:
    return {"items": []}
