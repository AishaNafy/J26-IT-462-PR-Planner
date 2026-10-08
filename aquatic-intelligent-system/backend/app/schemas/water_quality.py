from pydantic import BaseModel


class WaterQualityCreate(BaseModel):
    aquarium_id: str
    temperature_c: float
    ph: float
    turbidity: float
    tds: float


class WaterQualityRead(WaterQualityCreate):
    id: str
    recorded_at: str
    status: str
