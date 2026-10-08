from dataclasses import dataclass
from typing import Optional


@dataclass
class DiseaseCase:
    id: str
    fish_id: str
    diagnosis: str
    confidence: float
    detected_at: str
    severity: str = "moderate"
    created_at: Optional[str] = None
