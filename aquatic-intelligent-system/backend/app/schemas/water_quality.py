from typing import Optional

from pydantic import BaseModel, Field


class WaterQualityCreate(BaseModel):
    aquarium_id: str = "tank-01"
    device_id: str = "esp32-node-1"
    temperature_c: Optional[float] = Field(default=None, ge=-10, le=60)
    ph: Optional[float] = Field(default=None, ge=0, le=14)
    turbidity: Optional[float] = Field(default=None, ge=0, le=100, description="Relative turbidity index, not NTU")
    tds: Optional[float] = Field(default=None, ge=0)


class WaterQualityRead(WaterQualityCreate):
    id: str
    recorded_at: str
    status: str
