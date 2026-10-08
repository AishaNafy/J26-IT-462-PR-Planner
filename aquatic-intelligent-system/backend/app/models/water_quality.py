from dataclasses import dataclass
from typing import Optional


@dataclass
class WaterQualityRecord:
    id: str
    aquarium_id: str
    temperature_c: float
    ph: float
    turbidity: float
    tds: float
    recorded_at: str
    status: str = "ok"
    created_at: Optional[str] = None
