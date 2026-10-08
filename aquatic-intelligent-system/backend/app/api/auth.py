from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login() -> dict:
    return {"message": "Authentication endpoint placeholder"}
