from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.ml.disease.image_model import get_classifier

router = APIRouter(prefix="/disease-detection", tags=["disease-detection"])
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.get("/")
def list_disease_cases() -> dict:
    return {"items": []}


@router.post("/predict-image")
async def predict_image(image: UploadFile = File(...)) -> dict:
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image exceeds the 10 MB limit")
    try:
        return await run_in_threadpool(get_classifier().predict, data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
