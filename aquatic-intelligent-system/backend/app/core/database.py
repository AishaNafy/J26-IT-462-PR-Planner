from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import MONGODB_URL

client = AsyncIOMotorClient(MONGODB_URL)
database = client.get_default_database()


def get_database():
    return database
