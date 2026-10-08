from fastapi import APIRouter

router = APIRouter(prefix="/water-quality", tags=["water-quality"])


@router.get("/")
def list_water_quality() -> dict:
    return {"items": []}
