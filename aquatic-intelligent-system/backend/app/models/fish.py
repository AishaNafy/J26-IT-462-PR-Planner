from dataclasses import dataclass
from typing import Optional


@dataclass
class Fish:
    id: str
    aquarium_id: str
    name: str
    species: str
    age_months: int
    length_cm: float
    health_status: str
    created_at: Optional[str] = None
