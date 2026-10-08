from fastapi import APIRouter

router = APIRouter(prefix="/fish-population", tags=["fish-population"])


@router.get("/")
def list_population() -> dict:
    return {"items": []}
