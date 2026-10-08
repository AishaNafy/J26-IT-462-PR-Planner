from dataclasses import dataclass
from typing import Optional


@dataclass
class Aquarium:
    id: str
    name: str
    volume_liters: int
    environment_type: str
    owner_id: str
    created_at: Optional[str] = None
