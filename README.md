# ☀️ Solar Tracker — Dual-Axis ML + Physics Precision Tracker

A full-stack 2-axis solar tracking system for **Bhubaneswar, Odisha (20.28°N, 85.78°E)** combining real-time 3D simulation, a physics-grounded ML backend, and a live SCADA-style analytics dashboard.

---

## 📸 Screenshots

| 3D Simulator | Analytics Dashboard |
|---|---|
| Real-time Three.js scene with sun arc, panel tracking, and environment controls | Live SCADA-style power, irradiance, thermal, and tracking charts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BROWSER (Two Tabs)                       │
│                                                             │
│  deepseek_html_*.html          analytics.html               │
│  ┌──────────────────┐          ┌──────────────────────┐     │
│  │  3D Simulator    │ ──────▶  │  Power Analytics     │     │
│  │  Three.js scene  │localStorage│  Chart.js dashbd   │     │
│  │  Environment ctrl│          │  KPI cards + CSV exp │     │
│  └────────┬─────────┘          └──────────────────────┘     │
│           │ POST /predict (2s)                               │
└───────────┼─────────────────────────────────────────────────┘
            ▼
┌──────────────────────────────┐
│   Flask API  (port 5050)     │
│   api_server.py              │
│   • SPA tilt fallback        │
│   • ML hybrid inference      │
│   • Physics anomaly detect   │
│   • NOCT thermal model       │
│   • Beer-Lambert DNI/GHI     │
└──────────────────────────────┘
```

---

## ✨ Features

### 🔭 3D Solar Simulation (`deepseek_html_*.html`)
- **Three.js r128** real-time scene — dual-axis panel, sun arc, clouds, stars
- **Diurnal sun model** — elevation & azimuth computed from SPA equations for Bhubaneswar
- **Beer-Lambert irradiance** — GHI auto-computed from sun elevation + cloud cover (Kaplanis model)
- **NOCT thermal model** — panel cell temperature from ambient + wind + irradiance
- **Wind stow mode** — panel auto-parks flat at ≥15 m/s to prevent mechanical damage
- **Night mode** — hard GHI/DNI/power lock to 0 W when sun below horizon
- **Auto/Manual modes** — direct sun tracking or manual azimuth/elevation override
- **ML API integration** — hybrid ML+SPA tilt prediction when backend is online
- **Cross-tab broadcast** — publishes state to analytics page via `localStorage`

### 📊 Analytics Dashboard (`analytics.html`)
- **5 live Chart.js charts**: Power vs Time, Performance Ratio, Tracking Error, Irradiance (GHI/DNI), Thermal Profile
- **4 KPI cards**: Power output, Irradiance & environment, Kinematics, Efficiency derating
- **Anomaly banners**: typed fault alerts (efficiency drop, tracking fault, combined fault, thermal limit)
- **CSV export** — buffer up to 120 data points, export as timestamped `.csv`
- **Zero polling** — event-driven via `localStorage` storage listener

### 🤖 ML Backend (`api_server.py`)
- **Hybrid prediction**: 60–90% ML weight + SPA fallback, cloud-adaptive blending
- **Models**: Random Forest & Gradient Boosting for GHI and optimal tilt
- **Physics-gated anomaly detection** — three strict rules (no false positives at night/twilight):
  - `EFFICIENCY_DROP` — actual power < 80% of theoretical (DNI ≥ 50 W/m²)
  - `TRACKING_FAULT` — AOI > 5° in clear sky (cloud < 30%)
  - `ENVIRONMENTAL_FAULT` — panel temperature > 85°C
  - `COMBINED_FAULT` — multiple simultaneous conditions
- **True AOI calculation** — spherical dot-product panel normal vs sun vector
- **Performance ratio** — real-time energy efficiency metric

### 🧪 ML Pipeline (`solar_ml_pipeline.py`)
- Trains on **43,847 rows (2020–2024) + 8,759 rows (2025)** of real NASA POWER data
- Falls back to **physics-correct synthetic data** (SPA + Beer-Lambert + NOCT) if CSVs are absent
- Feature engineering: cyclic time encoding, GHI/DNI lag features (1h, 2h, 3h, 24h)
- Moving-window median outlier removal
- Physics-labeled anomaly classifier (no unsupervised IsolationForest false positives)
- Exports: `best_ghi_model.pkl`, `best_tilt_model.pkl`, `anomaly_model.pkl`, scalers, `features.json`

---

## 📁 Project Structure

```
solar-tracker/
│
├── deepseek_html_20260423_9e251f.html   # 3D simulator (main entry point)
├── analytics.html                       # SCADA analytics dashboard
├── api_server.py                        # Flask ML backend (port 5050)
├── solar_ml_pipeline.py                 # Model training pipeline
├── test_api.py                          # API unit tests (5 physics tests)
├── start_server.bat                     # One-click Windows launcher
│
├── features.json                        # Feature list for ML inference
├── evaluation_summary.csv               # Model RMSE/MAE/R² comparison
│
├── Sun_elevation_and_optimal_tilt_angle_Solar_Data_2020_2024.csv
├── Sun_elevation_and_optimal_tilt_angle_Solar_Data_2025.csv
├── POWER_Point_Hourly_20250101_20251231_020d28N_085d78E_UTC.csv
│
├── best_ghi_model.pkl                   # (generated) GHI prediction model
├── best_tilt_model.pkl                  # (generated) Tilt angle model
├── anomaly_model.pkl                    # (generated) Anomaly classifier
├── scaler_X.pkl / scaler_ghi.pkl / scaler_tilt.pkl  # (generated)
└── fig1_prediction_vs_actual.png        # (generated) Evaluation plot
```

---

## 🚀 Quick Start

### Windows (One-Click)

```bat
start_server.bat
```

This will:
1. Kill any stale processes on ports 5050 / 8080
2. Install Python dependencies
3. Start the Flask ML API on port 5050
4. Start an HTTP server on port 8080
5. Open both pages in your browser automatically

### Manual Setup

**1. Install dependencies**
```bash
pip install flask flask-cors numpy scikit-learn pandas
```

**2. Train the ML models** *(first run only, ~2–3 min)*
```bash
python solar_ml_pipeline.py
```

**3. Start the ML API backend**
```bash
python api_server.py
```

**4. Serve the frontend** *(in a separate terminal)*
```bash
python -m http.server 8080
```

**5. Open in browser**
- **3D Simulator**: http://localhost:8080/deepseek_html_20260423_9e251f.html
- **Analytics Dashboard**: http://localhost:8080/analytics.html
- **API Health Check**: http://localhost:5050/health

---

## 🔌 API Reference

### `POST /predict`

Predicts optimal panel tilt and GHI, with anomaly detection.

**Request body:**
```json
{
  "hour": 12.5,
  "month": 4,
  "doy": 113,
  "sunElevation": 65.0,
  "sunAzimuth": 170.0,
  "temp": 28.0,
  "wind": 4.0,
  "irradiance": 900.0,
  "cloud": 10.0
}
```

**Response:**
```json
{
  "tilt_deg": 25.4,
  "hybrid_tilt": 26.1,
  "spa_tilt": 25.0,
  "source": "hybrid_ml_spa",
  "ghi_wm2": 887.3,
  "dni_wm2": 712.1,
  "panel_temp": 52.4,
  "temp_derating_pct": 10.96,
  "theoretical_power_w": 268.5,
  "performance_ratio": 94.2,
  "is_anomaly": false,
  "anomaly_type": "NORMAL",
  "aoi_deg": 1.32,
  "is_stow": false,
  "is_night": false
}
```

### `GET /health`
Returns server status, model load state, and configuration.

### `GET /ping`
Lightweight connectivity check.

---

## 🧪 Running Tests

```bash
python test_api.py
```

Tests validate:
- **T1** Night → no false-positive anomaly
- **T2** Wind stow → no anomaly
- **T3** Clear midday → correct GHI and performance ratio
- **T4** Cloudy twilight (low DNI) → no false-positive anomaly
- **T5** Response schema — `anomaly_type` and `aoi_deg` present

---

## 🔬 Physics Models

| Model | Formula | Reference |
|---|---|---|
| Sun elevation | Spherical law of cosines (SPA) | NREL Solar Position Algorithm |
| Air mass | Kasten-Young formula | *Solar Energy* 1989 |
| GHI clear-sky | Bird simplified model | NREL TR-215-2537 |
| Cloud attenuation | Kaplanis model: `1 − 0.72 × cf³·²` | Kaplanis 2006 |
| DNI from GHI | Beer-Lambert decomposition + 50/50 blend | Standard practice |
| Panel temperature | NOCT model: `T_cell = T_amb + (NOCT−20)/800 × GHI − wind_cool` | IEC 61215 |
| Optimal tilt | `tilt = 90° − sun_elevation` | Direct normal tracking |
| AOI | Spherical dot-product: panel normal · sun vector | ASHRAE |

---

## 📈 Model Performance

| Model | Target | RMSE | MAE | R² |
|---|---|---|---|---|
| Random Forest | GHI (W/m²) | — | — | — |
| Gradient Boosting | GHI (W/m²) | — | — | — |
| Random Forest | Tilt (°) | — | — | — |
| Gradient Boosting | Tilt (°) | — | — | — |

*Run `solar_ml_pipeline.py` to populate `evaluation_summary.csv` with your trained model metrics.*

---

## 🛠️ Configuration

Key constants in `api_server.py`:

```python
LAT          = 20.28    # Bhubaneswar latitude (°N)
PANEL_AREA   = 1.6      # m²
EFF_STC      = 0.20     # Panel efficiency at STC (20%)
TEMP_COEFF   = 0.004    # Power loss per °C above 25°C
NOCT         = 44.0     # Nominal Operating Cell Temperature (°C)
WIND_STOW_MS = 15.0     # Wind stow threshold (m/s)
PORT         = 5050
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | ≥ 3.0 | REST API backend |
| flask-cors | ≥ 4.0 | Cross-origin requests |
| scikit-learn | ≥ 1.3 | Random Forest, Gradient Boosting |
| numpy | ≥ 1.24 | Numerical computation |
| pandas | ≥ 2.0 | Data loading and feature engineering |
| Three.js | r128 | 3D WebGL rendering (CDN) |
| Chart.js | 4.4.0 | Analytics charts (CDN) |

---

## 🗺️ Roadmap

- [ ] Real-time weather API integration (Open-Meteo / IMD)
- [ ] LSTM time-series model for improved GHI forecasting
- [ ] Multi-panel array support
- [ ] Export to InfluxDB / Grafana
- [ ] Mobile-responsive analytics layout
- [ ] Docker deployment (`docker-compose up`)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **NASA POWER** — Meteorological data for Bhubaneswar (20.28°N, 85.78°E)
- **NREL** — Solar Position Algorithm and Bird clear-sky model
- **Three.js** — WebGL 3D rendering engine
- **scikit-learn** — ML model training and evaluation
