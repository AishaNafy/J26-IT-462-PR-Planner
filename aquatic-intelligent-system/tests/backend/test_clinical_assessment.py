from app.services.clinical_assessment import combine_assessment
from app.services.esp32_serial import parse_firmware_line
from app.services.water_quality_service import evaluate_reading


def test_firmware_reading_line_is_parsed_and_evaluated():
    parsed = parse_firmware_line("Temp: 25.12 | pH: 7.10 (2.480 V) | Turbidity: 8 idx (4.000 V)")

    assert parsed == {"temperature_c": 25.12, "ph": 7.1, "turbidity": 8.0}
    assert evaluate_reading(parsed)["stage"] == "healthy"


def test_invalid_sensor_is_reported_as_fault_and_moderate():
    parsed = parse_firmware_line("Temp: n/a | pH: 7.10 (2.480 V) | Turbidity: 8 idx (4.000 V)")
    result = evaluate_reading(parsed)

    assert "temperature_c" in result["faults"]
    assert result["stage"] == "moderate"


def test_all_invalid_firmware_values_are_stored_as_sensor_faults():
    parsed = parse_firmware_line(
        "Temp: n/a | pH: 19.47 (0.213 V) [INVALID] | Turbidity: 100 idx (0.213 V) [INVALID]"
    )

    assert parsed == {"temperature_c": None, "ph": None, "turbidity": None}
    assert evaluate_reading(parsed)["faults"] == ["temperature_c", "ph", "turbidity"]


def test_healthy_requires_fish_and_all_three_normal_sensors():
    reading = {"temperature_c": 25.0, "ph": 7.0, "turbidity": 10.0}
    evaluated = {**reading, **evaluate_reading(reading)}
    prediction = {"is_fish": True, "label": "Healthy", "confidence": 0.9}

    assert combine_assessment(prediction, evaluated)["stage"] == "healthy"


def test_warning_readings_include_preferred_and_critical_ranges():
    result = evaluate_reading({"temperature_c": 31.0, "ph": 6.0, "turbidity": 35.0})

    assert result["stage"] == "moderate"
    assert result["warnings"] == ["temperature_c", "ph", "turbidity"]
    assert result["parameters"]["ph"]["preferred_min"] == 6.5
    assert result["parameters"]["ph"]["critical_min"] == 5.5
    assert result["parameters"]["turbidity"]["unit"] == "relative index"


def test_critical_sensor_result_escalates_fish_assessment():
    reading = {"temperature_c": 35.0, "ph": 7.1, "turbidity": 8.0}
    evaluated = {**reading, **evaluate_reading(reading)}
    prediction = {"is_fish": True, "label": "Healthy", "confidence": 0.9}

    result = combine_assessment(prediction, evaluated)

    assert result["stage"] == "critical"
    assert result["alert"] is True
    assert "temperature_c" in result["water_reading"]["critical"]


def test_non_fish_image_is_not_given_a_health_stage():
    result = combine_assessment({"is_fish": False, "label": None}, None)

    assert result["stage"] == "not_a_fish"
    assert result["alert"] is not True


def test_disease_fields_name_fish_class_and_disease_type():
    from app.ml.disease.image_model import FishDiseaseClassifier

    category, disease_name, label = FishDiseaseClassifier._disease_fields(
        "Fish_Bacterial diseases - Aeromoniasis"
    )

    assert category == "Bacterial"
    assert disease_name == "Aeromoniasis"
    assert label == "Bacterial diseases - Aeromoniasis"
    assert FishDiseaseClassifier._disease_fields("Fish_Healthy Fish")[1] == "No visible disease detected"