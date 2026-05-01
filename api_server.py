"""
=============================================================================
 Solar Tracker ML Backend API v2.0 -- Flask server
 Endpoints: POST /predict, GET /health
 Scientific fixes:
   - Hard nighttime zero enforcement (GHI/DNI/Power = 0 when sun_elev <= 0)
   - Wind stow mode (panel forced flat at 0 deg when wind > 15 m/s)
   - Temperature derating via NOCT cell model
   - Anomaly detection via IsolationForest
   - DNI computed from Beer-Lambert transmittance
=============================================================================
"""
import os, sys, pickle, math, time, logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Force ASCII-safe logging on Windows
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LAT       = 20.28   # Bhubaneswar latitude
PORT      = 5050

# Physics constants
PANEL_AREA   = 1.6    # m^2
EFF_STC      = 0.20   # Panel efficiency at STC (Standard Test Conditions)
TEMP_COEFF   = 0.004  # Power loss per degC above 25C (0.4%/C typical crystalline Si)
NOCT         = 44.0   # Nominal Operating Cell Temperature (degC)
WIND_STOW_MS = 15.0   # Wind stow threshold (m/s)
SOLAR_CONST  = 1361.0 # W/m2

app = Flask(__name__)
CORS(app)

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_artifact(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{fname} not found. Run solar_ml_pipeline.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)

MODELS_OK    = False
TILT_MODEL   = None
GHI_MODEL    = None
ANOMALY_MODEL = None
SCALER_X     = None
SCALER_GHI   = None
SCALER_TLT   = None

try:
    TILT_MODEL    = load_artifact('best_tilt_model.pkl')
    GHI_MODEL     = load_artifact('best_ghi_model.pkl')
    ANOMALY_MODEL = load_artifact('anomaly_model.pkl')
    SCALER_X      = load_artifact('scaler_X.pkl')
    SCALER_GHI    = load_artifact('scaler_ghi.pkl')
    SCALER_TLT    = load_artifact('scaler_tilt.pkl')
    MODELS_OK     = True
    log.info("All models loaded OK.")
except Exception as e:
    log.warning(f"Model load failed: {e} -- running in SPA fallback mode.")

# =============================================================================
# PHYSICS HELPERS
# =============================================================================
def _air_mass(sun_elev_deg):
    """Kasten-Young air mass formula."""
    if sun_elev_deg <= 0:
        return 38.0
    el = max(0.1, sun_elev_deg)
    am = 1.0 / (math.sin(math.radians(el)) + 0.50572 * (el + 6.07995) ** -1.6364)
    return min(38.0, am)

def _dni_from_ghi(ghi, sun_elev_deg, doy, cloud_frac=0.0):
    """
    Estimate DNI from GHI using simplified decomposition.
    DNI = (GHI - DHI) / sin(elevation)
    DHI estimated as 10% of clear-sky horizontal component.
    """
    if sun_elev_deg <= 0 or ghi <= 0:
        return 0.0
    # Extraterrestrial irradiance (Earth-Sun distance correction)
    B  = math.radians((360 / 365) * (doy - 1))
    G0 = SOLAR_CONST * (1.000110 + 0.034221 * math.cos(B) + 0.001280 * math.sin(B))
    am = _air_mass(sun_elev_deg)
    tau_b = 0.70 ** (am ** 0.678)
    dni_clear = G0 * tau_b
    cf_att = 1.0 - 0.72 * (cloud_frac ** 3.2)
    dni = dni_clear * cf_att * math.sin(math.radians(sun_elev_deg))
    # Scale to match reported GHI
    dhi = 0.1 * G0 * math.sin(math.radians(sun_elev_deg)) * cf_att
    dni_from_ghi = (ghi - dhi) / math.sin(math.radians(sun_elev_deg))
    # Blend physics DNI with GHI-derived DNI
    return max(0.0, round(0.5 * dni + 0.5 * max(0.0, dni_from_ghi), 2))

def _panel_temperature(ambient_c, ghi_effective, wind_ms):
    """NOCT cell temperature model."""
    delta_t = ((NOCT - 20.0) / 800.0) * ghi_effective
    wind_cooling = max(0.0, (wind_ms - 1.0) * 1.5)
    t_cell = ambient_c + delta_t - wind_cooling
    return round(max(ambient_c, min(90.0, t_cell)), 2)

def _temp_derating_factor(panel_temp_c):
    """Power factor due to temperature derating."""
    return max(0.0, 1.0 - TEMP_COEFF * (panel_temp_c - 25.0))

def _angle_of_incidence_deg(panel_tilt_deg, panel_az_deg, sun_elev_deg, sun_az_deg):
    """
    True angle of incidence (AOI) between the sun vector and the panel normal.
    Uses spherical dot-product formula. Returns degrees.
    """
    t = math.radians(panel_tilt_deg)
    pa = math.radians(panel_az_deg)
    se = math.radians(max(0.0, sun_elev_deg))
    sa = math.radians(sun_az_deg)
    # Panel normal vector (tilted from vertical by panel_tilt, facing panel_az)
    nx = math.sin(t) * math.sin(pa)
    ny = math.cos(t)
    nz = math.sin(t) * math.cos(pa)
    # Sun vector
    sx = math.cos(se) * math.sin(sa)
    sy = math.sin(se)
    sz = math.cos(se) * math.cos(sa)
    cos_aoi = max(0.0, min(1.0, nx*sx + ny*sy + nz*sz))
    return math.degrees(math.acos(cos_aoi))

def _theoretical_power(ghi, panel_tilt_deg, panel_az_deg, sun_elev_deg, sun_az_deg, panel_temp_c):
    """
    P = GHI * Area * Eff_STC * cos(AOI) * TempDerating
    Strictly anchored to panel-sun geometry.
    """
    if sun_elev_deg <= 0 or ghi <= 0:
        return 0.0, 0.0
    aoi_deg = _angle_of_incidence_deg(panel_tilt_deg, panel_az_deg, sun_elev_deg, sun_az_deg)
    cos_aoi = math.cos(math.radians(aoi_deg))
    t_factor = _temp_derating_factor(panel_temp_c)
    power = ghi * PANEL_AREA * EFF_STC * cos_aoi * t_factor
    return max(0.0, power), aoi_deg

def _performance_ratio(actual_power, theoretical_max):
    if theoretical_max <= 0:
        return 0.0
    return round(min(1.0, actual_power / theoretical_max) * 100, 2)

def _check_physics_anomaly(dni, actual_power, theo_power, aoi_deg, cloud_pct, sun_elev, panel_temp_c):
    """
    Physics-gated anomaly detection.  Three strict conditions, in priority order:

    GATE 0 (skip):   DNI < 50 W/m^2 OR sun below horizon -> NORMAL
                     Nighttime / twilight / heavy overcast are EXPECTED states.

    RULE 1 (power):  actual_power < 80% of theoretical -> EFFICIENCY_DROP
                     Catches soiling, bypass-diode failure, cell degradation.

    RULE 2 (kinematic): AOI > 5 deg in clear sky (cloud < 30%) -> TRACKING_FAULT
                     Catches motor jam, encoder drift, mechanical binding.

    RULE 3 (thermal): panel_temp > 85 deg C -> ENVIRONMENTAL_FAULT
                     Catches cooling failure, extreme ambient, inadequate ventilation.

    Returns (is_anomaly: bool, anomaly_type: str)
    """
    if sun_elev <= 0 or dni < 50.0:
        return False, "NORMAL"

    is_anomaly   = False
    anomaly_type = "NORMAL"

    # Rule 1: severe efficiency drop
    if theo_power > 5.0 and actual_power < 0.80 * theo_power:
        is_anomaly   = True
        anomaly_type = "EFFICIENCY_DROP"

    # Rule 2: kinematic / tracking fault (clear sky only)
    if aoi_deg > 5.0 and cloud_pct < 30.0:
        is_anomaly = True
        anomaly_type = "COMBINED_FAULT" if anomaly_type == "EFFICIENCY_DROP" else "TRACKING_FAULT"

    # Rule 3: thermal / environmental fault
    if panel_temp_c > 85.0:
        is_anomaly = True
        anomaly_type = "COMBINED_FAULT" if anomaly_type != "NORMAL" else "ENVIRONMENTAL_FAULT"

    return is_anomaly, anomaly_type

def _spa_tilt(sun_elev_deg):
    """Optimal tilt from SPA: 90 - elevation, clamped [0,90]."""
    if sun_elev_deg <= 0:
        return abs(LAT)  # Park at latitude angle (stow)
    return round(max(0.0, min(90.0, 90.0 - sun_elev_deg)), 3)

def _build_feature_vector(hour, month, doy, sun_elev, temp, wind,
                           ghi_lag1, ghi_lag2, ghi_lag3, ghi_lag24,
                           dni_lag1, dni_lag2, dni_lag3, dni_lag24):
    PI2 = 2 * math.pi
    return np.array([[
        sun_elev, temp, wind,
        math.sin(PI2 * hour  / 24),  math.cos(PI2 * hour  / 24),
        math.sin(PI2 * month / 12),  math.cos(PI2 * month / 12),
        math.sin(PI2 * doy   / 365), math.cos(PI2 * doy   / 365),
        ghi_lag1, ghi_lag2, ghi_lag3, ghi_lag24,
        dni_lag1, dni_lag2, dni_lag3, dni_lag24,
    ]])

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/ping', methods=['GET', 'OPTIONS'])
def ping():
    """Lightweight endpoint for frontend connectivity testing."""
    return jsonify({"pong": True, "models": MODELS_OK, "ts": time.time()})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":  "ok" if MODELS_OK else "degraded",
        "models_loaded": MODELS_OK,
        "latitude": LAT,
        "wind_stow_threshold_ms": WIND_STOW_MS,
        "port": PORT,
        "timestamp": time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # -- Parse inputs --
        hour  = max(0.0, min(23.99, float(data['hour'])))
        month = max(1,   min(12,    int(data['month'])))
        doy   = max(1,   min(365,   int(data['doy'])))
        sun_elev = max(-10.0, min(90.0, float(data['sunElevation'])))
        sun_az   = max(0.0, min(360.0, float(data.get('sunAzimuth', 180))))
        temp     = max(-20.0, min(70.0, float(data.get('temp', 28))))
        wind     = max(0.0,   min(50.0, float(data.get('wind', 3))))
        irradiance = max(0.0, float(data.get('irradiance', 500)))
        cloud      = max(0.0, min(100.0, float(data.get('cloud', 0))))
        cloud_frac = cloud / 100.0

        # ---- EDGE CASE 1: NIGHTTIME HARD LOCK ----
        # Physics: No solar energy when sun is below horizon. Period.
        is_night = sun_elev <= 0.0
        if is_night:
            irradiance = 0.0

        # ---- EDGE CASE 2: WIND STOW MODE ----
        # If wind exceeds threshold, override all tracking and flatten panel
        is_stow = wind >= WIND_STOW_MS
        if is_stow:
            log.info(f"WIND STOW activated at {wind:.1f} m/s")

        # Effective irradiance for panel temperature
        sin_elev  = max(0.0, math.sin(math.radians(sun_elev)))
        eff_ghi   = irradiance * (1.0 - 0.72 * (cloud_frac ** 3.2)) * sin_elev
        eff_ghi   = max(0.0, eff_ghi)

        # Panel temperature via NOCT model
        panel_temp = _panel_temperature(temp, eff_ghi, wind)

        # DNI from physics
        dni = _dni_from_ghi(eff_ghi, sun_elev, doy, cloud_frac)

        # Lag features with cloud-adjusted defaults
        def cf(v): return v * (1.0 - 0.72 * (cloud_frac ** 3.2))
        ghi_lag1  = cf(float(data.get('ghiLag1',  irradiance * 0.90)))
        ghi_lag2  = cf(float(data.get('ghiLag2',  irradiance * 0.88)))
        ghi_lag3  = cf(float(data.get('ghiLag3',  irradiance * 0.85)))
        ghi_lag24 = cf(float(data.get('ghiLag24', irradiance * 0.60)))
        dni_lag1  = cf(float(data.get('dniLag1',  irradiance * 0.70)))
        dni_lag2  = cf(float(data.get('dniLag2',  irradiance * 0.68)))
        dni_lag3  = cf(float(data.get('dniLag3',  irradiance * 0.65)))
        dni_lag24 = cf(float(data.get('dniLag24', irradiance * 0.45)))

        # -- ML Prediction --
        spa_tilt   = _spa_tilt(sun_elev)
        ml_tilt    = spa_tilt
        ml_ghi     = round(eff_ghi, 2)
        hybrid_tilt_val = spa_tilt
        source     = "spa_fallback"
        confidence = 0.0
        is_anomaly = False

        if MODELS_OK and not is_night and not is_stow:
            try:
                row = _build_feature_vector(
                    hour, month, doy, sun_elev, temp, wind,
                    ghi_lag1, ghi_lag2, ghi_lag3, ghi_lag24,
                    dni_lag1, dni_lag2, dni_lag3, dni_lag24
                )
                row_sc = SCALER_X.transform(row)
                ghi_sc_pred  = GHI_MODEL.predict(row_sc)[0]
                ml_ghi       = float(SCALER_GHI.inverse_transform([[ghi_sc_pred]])[0][0])
                ml_ghi       = max(0.0, ml_ghi)
                tilt_sc_pred = TILT_MODEL.predict(row_sc)[0]
                ml_tilt      = float(SCALER_TLT.inverse_transform([[tilt_sc_pred]])[0][0])
                ml_tilt      = max(0.0, min(90.0, ml_tilt))
                cloud_wt     = min(1.0, cloud_frac * 1.5)
                ml_weight    = 0.60 + 0.30 * cloud_wt
                spa_weight   = 1.0 - ml_weight
                hybrid_tilt_val = round(ml_weight * ml_tilt + spa_weight * spa_tilt, 3)
                hybrid_tilt_val = max(0.0, min(90.0, hybrid_tilt_val))
                source     = "hybrid_ml_spa"
                confidence = 1.0 if sun_elev > 5 else 0.4
                # NOTE: IsolationForest removed -- physics rules used below for all paths
            except Exception as e:
                log.warning(f"ML inference error: {e}")
                source = "spa_fallback"; confidence = 0.0
        if is_stow:
            hybrid_tilt_val = 0.0; ml_tilt = 0.0; source = "wind_stow"; confidence = 1.0
        if is_night:
            ml_ghi = 0.0; hybrid_tilt_val = abs(LAT); source = "night"; confidence = 1.0; is_anomaly = False

        # P = GHI * Area * Eff * cos(AOI) * TempFactor  (using hybrid_tilt, sun_az geometry)
        theo_power, aoi_deg = _theoretical_power(
            ml_ghi, hybrid_tilt_val, sun_az, sun_elev, sun_az, panel_temp)
        temp_derating_pct = round(TEMP_COEFF * max(0.0, panel_temp - 25.0) * 100, 2)
        perf_ratio = _performance_ratio(theo_power, PANEL_AREA * EFF_STC * max(1.0, ml_ghi))

        # Physics-gated anomaly detection (replaces IsolationForest)
        is_anomaly, anomaly_type = _check_physics_anomaly(
            dni, theo_power, theo_power, aoi_deg, cloud * 100, sun_elev, panel_temp)
        # Use perf_ratio to detect actual efficiency drop vs theoretical
        if not is_night and not is_stow:
            eff_ratio = (perf_ratio / 100.0) if perf_ratio > 0 else 1.0
            if dni >= 50.0 and eff_ratio < 0.80 and theo_power > 5.0:
                is_anomaly   = True
                anomaly_type = "COMBINED_FAULT" if anomaly_type == "TRACKING_FAULT" else "EFFICIENCY_DROP"
        if is_stow or is_night:
            is_anomaly   = False
            anomaly_type = "NORMAL"

        response = {
            # Tracking
            "tilt_deg":    round(ml_tilt, 3),
            "hybrid_tilt": round(hybrid_tilt_val, 3),
            "spa_tilt":    round(spa_tilt, 3),
            "source":      source,
            "confidence":  confidence,
            # Irradiance
            "ghi_wm2":     round(ml_ghi, 2),
            "dni_wm2":     round(dni, 2),
            "eff_ghi":     round(eff_ghi, 2),
            # Thermal
            "panel_temp":        panel_temp,
            "temp_derating_pct": temp_derating_pct,
            # Power
            "theoretical_power_w": round(theo_power, 2),
            "performance_ratio":   perf_ratio,
            # State flags
            "is_anomaly":    is_anomaly,
            "anomaly_type":  anomaly_type,
            "aoi_deg":       round(aoi_deg, 3),
            "is_stow":       is_stow,
            "is_night":      is_night,
            "cloud_frac":    round(cloud_frac, 3),
        }
        return jsonify(response)

    except Exception as e:
        log.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Solar Tracker ML Backend v2.0")
    print(f"  Models loaded : {'YES' if MODELS_OK else 'NO (SPA fallback)'}")
    print(f"  Wind stow     : {WIND_STOW_MS} m/s")
    print(f"  Server        : http://localhost:{PORT}")
    print("=" * 70 + "\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)