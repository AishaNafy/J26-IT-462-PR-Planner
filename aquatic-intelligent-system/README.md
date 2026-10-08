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

1. Copy `.env.example` to `.env` and adjust values.
2. Start the stack:

```bash
docker-compose up --build
```

3. Access the API at `http://localhost:8000`.
4. Run the Flutter app using `flutter pub get` and `flutter run`.

## Features

- Smart fish feeding schedule
- Water quality monitoring
- Fish population analytics
- Disease detection and alerting
- Dashboard overview for aquarium health
- Notification and alert system

## Environment Variables

See `.env.example` for required configuration values.
