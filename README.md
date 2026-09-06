# RoboSentinel

**Explainable, unsupervised anomaly detection for robot telemetry.**

RoboSentinel watches multivariate telemetry such as motor current, motor temperature, vibration, battery voltage, velocity error, and control-loop latency. It learns a normal operating baseline, detects abnormal behavior without requiring labeled failure examples, and ranks the sensors most responsible for each alert.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![ML](https://img.shields.io/badge/ML-IsolationForest-orange)
![CI](https://img.shields.io/badge/tests-pytest-success)

## Why I built it

Robots produce large streams of sensor data, but many failures are **correlated events** rather than a single obvious threshold violation. A dragging drivetrain, for example, can simultaneously increase current draw, increase velocity error, and sag battery voltage. RoboSentinel is designed to catch those multivariate patterns and produce an explanation that a human can inspect.

## Highlights

- **Hybrid anomaly detector:** Isolation Forest + robust median/MAD statistics
- **Time-series feature engineering:** deltas, EMA residuals, rolling local z-scores
- **Physics-inspired interactions:** current/voltage, tracking stress, thermal load, vibration-current coupling
- **Root-cause attribution:** ranks the raw sensors driving each alert
- **Fault simulator:** generates correlated drivetrain, thermal, brownout, vibration, and control-loop faults
- **FastAPI service:** analyze telemetry via REST or CSV upload
- **Zero-framework dashboard:** interactive browser UI served directly by FastAPI
- **CLI + Docker + tests + GitHub Actions**

## Architecture

```text
                ┌─────────────────────┐
CSV / simulator │ raw robot telemetry │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │ feature engineering │
                │ delta / EMA / z /   │
                │ cross-signal terms  │
                └──────────┬──────────┘
                           │
               ┌───────────┴───────────┐
               v                       v
      ┌────────────────┐      ┌─────────────────┐
      │ IsolationForest│      │ robust MAD score│
      └───────┬────────┘      └────────┬────────┘
              └───────────┬────────────┘
                          v
                 ┌──────────────────┐
                 │ ensemble anomaly │
                 │      score       │
                 └────────┬─────────┘
                          v
                 ┌──────────────────┐
                 │ root-cause rank  │
                 └────────┬─────────┘
                          v
             Dashboard / API / scored CSV
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"
pytest -q
uvicorn robosentinel.api:app --reload
```

Then open `http://127.0.0.1:8000`.

The dashboard automatically generates a synthetic robot run. Click **Generate Demo** for another run, or upload a CSV with the required sensor columns.

## CLI

Generate a dataset:

```bash
robosentinel simulate --rows 2000 --seed 42 --output telemetry.csv
```

Analyze it:

```bash
robosentinel analyze telemetry.csv --output scored_telemetry.csv
```

## Expected telemetry schema

```csv
timestamp_ms,motor_current_a,motor_temp_c,vibration_rms,battery_voltage_v,velocity_error,loop_latency_ms
0,21.3,51.2,0.41,12.11,0.02,8.9
20,21.8,51.4,0.40,12.08,0.03,9.1
```

`timestamp_ms` is optional for the detector itself. The six sensor columns are required by the default configuration.

## How the detector works

### 1. Learn a healthy baseline

The first portion of a run is treated as a baseline. If synthetic ground-truth labels exist, known faults are excluded from this baseline.

### 2. Engineer temporal features

Each raw signal produces:

- first difference
- exponentially weighted moving-average residual
- local rolling z-score

RoboSentinel also builds cross-signal interactions that model behavior such as drivetrain stress and voltage sag.

### 3. Score anomalies with an ensemble

An **Isolation Forest** captures multivariate patterns. A separate robust score uses the **median absolute deviation (MAD)** of each physical sensor. The two scores are combined to reduce dependence on any single method.

### 4. Explain each alert

For every sample, RoboSentinel computes the robust deviation of the original physical sensors and returns the top three likely contributors. This makes a detected anomaly inspectable instead of returning only a black-box score.

## API

### `GET /api/demo`

Generates, scores, and returns a synthetic robot run.

### `POST /api/analyze`

Upload a telemetry CSV as multipart form data.

### `GET /health`

Simple service health check.

Interactive OpenAPI docs are available at `/docs` when the server is running.

## Example research questions this repo can support

- How does a hybrid statistical/ML detector compare with fixed sensor thresholds?
- Which telemetry features are most useful for identifying drivetrain faults?
- How quickly can a fault be detected after onset?
- Can the detector generalize across different robot loads or gearing?
- What changes are needed for real-time streaming from a roboRIO, Arduino, ESP32, or ROS2 node?

## Roadmap

- [ ] WebSocket live-streaming telemetry endpoint
- [ ] Online/rolling baseline adaptation
- [ ] ROS2 subscriber integration
- [ ] SHAP-style feature attribution for engineered features
- [ ] Export alerts to Discord/Slack
- [ ] Benchmark against One-Class SVM and Local Outlier Factor
- [ ] Real robot dataset and detection-latency benchmark

## Tech stack

Python · NumPy · pandas · scikit-learn · FastAPI · vanilla JS/SVG · pytest · Docker · GitHub Actions

## License

MIT
