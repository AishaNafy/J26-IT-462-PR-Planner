from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/")
def dashboard_summary() -> dict:
    return {"aquariums": 0, "alerts": 0, "health_score": 100}
