from pydantic import BaseModel


class FishCreate(BaseModel):
    aquarium_id: str
    name: str
    species: str
    age_months: int
    length_cm: float
    health_status: str


class FishRead(FishCreate):
    id: str
