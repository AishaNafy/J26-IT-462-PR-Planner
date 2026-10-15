import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

APP_ENV = os.getenv("APP_ENV", "development")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/aquatic_intelligent")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DISEASE_MODEL_PATH = Path(os.getenv("DISEASE_MODEL_PATH", BASE_DIR.parent / "ml" / "trained_models" / "disease" / "fish_disease.onnx"))
DISEASE_LABELS_PATH = Path(os.getenv("DISEASE_LABELS_PATH", BASE_DIR.parent / "ml" / "trained_models" / "disease" / "labels.json"))
