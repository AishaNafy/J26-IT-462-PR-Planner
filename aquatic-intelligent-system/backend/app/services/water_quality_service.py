from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

LIMITS = {
    "temperature_c": {"min": 20.0, "max": 30.0, "critical_min": 18.0, "critical_max": 32.0, "unit": "°C"},
    "ph": {"min": 6.5, "max": 8.5, "critical_min": 5.5, "critical_max": 9.5, "unit": "pH"},
    "turbidity": {"min": 0.0, "max": 30.0, "critical_min": 0.0, "critical_max": 60.0, "unit": "relative index"},
}

EFFECTS = {
    "temperature_c": "Temperature affects fish metabolism, appetite, and oxygen demand.",
    "ph": "Out-of-range or rapidly changing pH can stress fish skin and gills.",
    "turbidity": "Suspended particles can reduce visibility and irritate gills; this is a relative index, not NTU.",
}

_readings: deque[dict[str, Any]] = deque(maxlen=500)
_lock = Lock()


def evaluate_reading(reading: dict[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    faults: list[str] = []
    warnings: list[str] = []
    critical: list[str] = []
    for name, limits in LIMITS.items():
        value = reading.get(name)
        if value is None:
            parameters[name] = {"value": None, "status": "fault", "message": f"{name} sensor is not reporting"}
            faults.append(name)
            continue

        if name == "turbidity":
            normal = value <= limits["max"]
            is_critical = value > limits["critical_max"]
        else:
            normal = limits["min"] <= value <= limits["max"]
            is_critical = value < limits["critical_min"] or value > limits["critical_max"]

        state = "normal" if normal else "critical" if is_critical else "warning"
        parameters[name] = {"value": value, "status": state, "message": EFFECTS[name]}
        if state == "critical":
            critical.append(name)
        elif state == "warning":
            warnings.append(name)

    if critical:
        stage = "critical"
    elif warnings or faults:
        stage = "moderate"
    else:
        stage = "healthy"
    return {"parameters": parameters, "faults": faults, "warnings": warnings, "critical": critical, "stage": stage}


class WaterQualityService:
    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        reading = {
            "id": str(uuid4()),
            "aquarium_id": payload.get("aquarium_id", "tank-01"),
            "device_id": payload.get("device_id", "esp32-node-1"),
            "temperature_c": payload.get("temperature_c"),
            "ph": payload.get("ph"),
            "turbidity": payload.get("turbidity"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        reading.update(evaluate_reading(reading))
        with _lock:
            _readings.append(reading)
        return reading

    def latest(self) -> dict[str, Any] | None:
        with _lock:
            return dict(_readings[-1]) if _readings else None

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with _lock:
            return [dict(reading) for reading in list(_readings)[-limit:]]


water_quality_service = WaterQualityService()
