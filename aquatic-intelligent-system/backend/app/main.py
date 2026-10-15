from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.disease_detection import router as disease_router
from app.api.feeding import router as feeding_router
from app.api.fish_population import router as fish_population_router
from app.api.water_quality import router as water_quality_router

app = FastAPI(title="Aquatic Intelligent System API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(feeding_router)
app.include_router(water_quality_router)
app.include_router(fish_population_router)
app.include_router(disease_router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "aquatic-intelligent-system"}


@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Aquatic Intelligent System API"}
