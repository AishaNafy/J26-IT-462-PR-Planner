from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core import config
from app.schemas.water_quality import WaterQualityCreate
from app.services.esp32_serial import serial_reader
from app.services.water_quality_service import LIMITS, water_quality_service

router = APIRouter(prefix="/water-quality", tags=["water-quality"])
_SERIAL_PORTS = ("/dev/cu.usbserial-", "/dev/cu.SLAB_USBtoUART", "/dev/ttyUSB", "/dev/ttyACM")


class SerialConnectRequest(BaseModel):
    port: str = Field(min_length=1)


def _available_ports() -> list[dict[str, str]]:
    from serial.tools import list_ports

    return [
        {"port": item.device, "description": item.description}
        for item in list_ports.comports()
        if item.device.startswith(_SERIAL_PORTS)
    ]


@router.get("/")
def list_water_quality(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"items": water_quality_service.history(limit)}


@router.get("/latest")
def latest_water_quality() -> dict:
    reading = water_quality_service.latest()
    if reading is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No ESP32 sensor readings received yet")
    return reading


@router.post("/readings", status_code=status.HTTP_201_CREATED)
def ingest_reading(payload: WaterQualityCreate, x_device_key: Optional[str] = Header(default=None)) -> dict:
    if config.DEVICE_API_KEY and x_device_key != config.DEVICE_API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-Device-Key")
    return water_quality_service.record(payload.model_dump())


@router.get("/serial/ports")
def serial_ports() -> dict:
    return {"items": _available_ports()}


@router.get("/serial/status")
def serial_status() -> dict:
    return serial_reader.status()


@router.post("/serial/connect")
def connect_serial(body: SerialConnectRequest) -> dict:
    available = {item["port"] for item in _available_ports()}
    if body.port not in available:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "ESP32 serial port is not available")
    try:
        return serial_reader.start(body.port)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/serial/disconnect")
def disconnect_serial() -> dict:
    return serial_reader.stop()


@router.get("/thresholds")
def water_thresholds() -> dict:
    return {
        "items": {
            name: {
                "safe_min": limits["min"],
                "safe_max": limits["max"],
                "critical_min": limits["critical_min"],
                "critical_max": limits["critical_max"],
                "unit": limits["unit"],
            }
            for name, limits in LIMITS.items()
        }
    }
