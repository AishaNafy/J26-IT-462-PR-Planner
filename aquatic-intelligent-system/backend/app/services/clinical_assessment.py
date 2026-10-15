from __future__ import annotations

from typing import Any


def combine_assessment(prediction: dict[str, Any], water_reading: dict[str, Any] | None) -> dict[str, Any]:
    if not prediction.get("is_fish"):
        return {
            **prediction,
            "stage": "not_a_fish",
            "reasons": ["No fish was confidently identified. Retake the photo with one fish clearly visible."],
            "sensor_faults": [],
            "water_reading": water_reading,
            "alert": False,
        }

    reasons: list[str] = []
    sensor_faults = ["temperature_c", "ph", "turbidity"] if water_reading is None else water_reading["faults"]
    if water_reading is None:
        reasons.append("No water-sensor readings received; health stage is provisional.")
        stage = "moderate"
    elif water_reading["stage"] == "critical":
        stage = "critical"
        reasons.extend(
            f"{name.replace('_', ' ').title()} is in the critical range."
            for name in water_reading["critical"]
        )
    else:
        if water_reading["warnings"]:
            reasons.extend(
                f"{name.replace('_', ' ').title()} is outside the preferred range."
                for name in water_reading["warnings"]
            )
        if sensor_faults:
            reasons.append("One or more water sensors are not reporting; check the probe and wiring.")
        if prediction.get("label") == "Abnormal":
            reasons.append("The image model found an abnormal visual pattern; this is not a disease diagnosis.")
        if prediction.get("confidence", 0) < 0.65:
            reasons.append("Image confidence is low; retake a clear photo and review the result manually.")
        stage = "moderate" if reasons else "healthy"

    if not reasons:
        reasons.append("Fish image appears healthy and all three reported water readings are in the preferred ranges.")
    return {
        **prediction,
        "stage": stage,
        "reasons": reasons,
        "sensor_faults": sensor_faults,
        "water_reading": water_reading,
        "alert": stage == "critical",
    }
