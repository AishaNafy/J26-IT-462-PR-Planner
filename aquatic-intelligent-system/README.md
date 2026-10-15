# Aquatic Intelligent System

A full-stack aquatic management platform for smart aquarium monitoring, feeding automation, fish population tracking, and disease intelligence.

## Architecture

- Frontend: Flutter mobile app
- Backend: FastAPI REST API
- Database: MongoDB
- IoT: ESP32 and sensor integrations
- ML: predictive models for feeding, water quality, fish population, and disease detection

## Project Structure

```text
aquatic-intelligent-system/
├── frontend/             # Flutter mobile app
├── backend/              # FastAPI backend
├── iot/                  # Device and sensor integrations
├── ml/                   # Data science and model artifacts
├── database/             # Seed and sample data
├── docs/                 # Architecture and API docs
├── tests/                # Cross-layer testing
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## Quick Start

### Fish Image Web Demo

Create the backend environment and install the API plus model runtime dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
```

Start FastAPI from the project root:

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

In a second terminal, start the Flutter Web app:

```bash
cd frontend
flutter pub get
flutter run -d chrome --web-port 5000
```

Open `http://localhost:5000`. The app uploads an image to `POST /disease-detection/predict-image`; the trained ONNX model is in `ml/trained_models/disease/`.

From the dashboard, open **Live water quality**, select the `/dev/cu.usbserial-*` ESP32 port, and connect. Close Arduino Serial Monitor first because USB serial can be owned by only one application at a time. The app polls temperature, pH, turbidity index, and probe faults. To use Flutter on a phone, pass the laptop's Wi-Fi IP as `--dart-define=API_BASE_URL=http://<laptop-ip>:8000`; the phone and laptop must be on the same LAN.

Prototype thresholds: preferred 20–30 °C, pH 6.5–8.5, relative turbidity index 0–30; critical <18 or >32 °C, pH <5.5 or >9.5, or turbidity >60. These are generic demonstration ranges, not species-specific veterinary limits. Turbidity is relative, not NTU. The three-stage rule is transparent: Critical for a valid hard-limit breach; Moderate for abnormal image, warning-range value, low image confidence, or any sensor fault; Healthy only when a fish is recognized and all three sensors are valid and in range. Sensor readings are not ML-trained because there are no labeled fish-outcome records for them.

The mobile camera currently captures a still photo, not a continuous video clip. Each photo is analyzed as one frame. Sensor readings are held in memory for the current backend process; they are cleared when the API restarts.

To retrain using the supplied dataset, install the training packages and run the command in [the disease model guide](ml/notebooks/disease/README.md). The local dataset is not stored in this repository.

## Features

- Smart fish feeding schedule
- Water quality monitoring
- Fish population analytics
- Disease detection and alerting
- Dashboard overview for aquarium health
- Notification and alert system

## Environment Variables

See `.env.example` for required configuration values.
