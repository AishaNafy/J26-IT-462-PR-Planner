from fastapi import APIRouter

router = APIRouter(prefix="/feeding", tags=["feeding"])


@router.get("/")
def list_feeding() -> dict:
    return {"items": []}
