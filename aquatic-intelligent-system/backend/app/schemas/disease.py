from pydantic import BaseModel


class DiseaseCreate(BaseModel):
    fish_id: str
    diagnosis: str
    confidence: float
    severity: str = "moderate"


class DiseaseRead(DiseaseCreate):
    id: str
    detected_at: str
