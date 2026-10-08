from pydantic import BaseModel


class FeedingCreate(BaseModel):
    aquarium_id: str
    scheduled_at: str
    amount_grams: float
    feed_type: str


class FeedingRead(FeedingCreate):
    id: str
    status: str
