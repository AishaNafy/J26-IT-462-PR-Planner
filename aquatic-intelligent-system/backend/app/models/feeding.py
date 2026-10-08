from dataclasses import dataclass
from typing import Optional


@dataclass
class FeedingEvent:
    id: str
    aquarium_id: str
    scheduled_at: str
    amount_grams: float
    feed_type: str
    status: str = "scheduled"
    created_at: Optional[str] = None
