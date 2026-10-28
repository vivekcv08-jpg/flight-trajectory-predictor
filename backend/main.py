"""
backend/main.py
===============
FastAPI Inference Server for Instant Flight Trajectory Predictor
Serves frontend at GET / and exposes:
  GET  /api/metrics  → benchmark metrics JSON
  POST /api/predict  → ML trajectory prediction
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_DIR  = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODELS_DIR   = BACKEND_DIR / "models"
JOBLIB_PATH  = MODELS_DIR / "trajectory_models.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Ensure backend package is importable for train_model + joblib unpickling
sys.path.insert(0, str(PROJECT_DIR))

# Import physics_feature_expansion so joblib can find it when unpickling
# (the FunctionTransformer was pickled with module=backend.physics)
from backend.physics import physics_feature_expansion, G, K_DRAG  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy-load models (with auto-training fallback)
# ---------------------------------------------------------------------------
_models:  dict = {}
_metrics: dict = {}


def _load_or_train():
    global _models, _metrics
    if not JOBLIB_PATH.exists() or not METRICS_PATH.exists():
        print("Models not found — running training pipeline...")
        from backend.train_model import train_and_export
        _models, _metrics = train_and_export()
    else:
        _models  = joblib.load(JOBLIB_PATH)
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    print(f"Loaded models: {list(_models.keys())}")


# ---------------------------------------------------------------------------
# ODE Physics (for ground-truth comparison)
# ---------------------------------------------------------------------------
def _eom(t, state, m):
    x, y, vx, vy = state
    v = math.hypot(vx, vy)
    k = K_DRAG / m
    return [vx, vy, -k * v * vx, -G - k * v * vy]


def _run_ode(v0: float, angle_deg: float, m: float):
    from scipy.integrate import solve_ivp
    theta  = math.radians(angle_deg)
    state0 = [0.0, 0.0, v0 * math.cos(theta), v0 * math.sin(theta)]

    def apex_event(t, s, m):  return s[3]
    apex_event.terminal  = False
    apex_event.direction = -1

    def ground_event(t, s, m): return s[1]
    ground_event.terminal  = True
    ground_event.direction = -1

    t0 = time.perf_counter()
    sol = solve_ivp(
        _eom, [0, 2000], state0, args=(m,),
        method='RK45', rtol=1e-6, atol=1e-8,
        events=[apex_event, ground_event],
        dense_output=False, max_step=1.0,
    )
    ode_ms = (time.perf_counter() - t0) * 1000.0

    xs = sol.y[0].tolist()
    ys = sol.y[1].tolist()

    if sol.t_events[1].size > 0:
        ye = sol.y_events[1][-1]
        ode_range        = float(ye[0])
        ode_impact_speed = float(math.hypot(ye[2], ye[3]))
    else:
        ode_range        = float(sol.y[0, -1])
        ode_impact_speed = float(math.hypot(sol.y[2, -1], sol.y[3, -1]))

    if sol.t_events[0].size > 0:
        ye_a = sol.y_events[0][-1]
        ode_alt = float(ye_a[1])
    else:
        ode_alt = float(np.max(sol.y[1]))

    return {
        "ode_range":        max(ode_range, 0.0),
        "ode_altitude":     max(ode_alt, 0.0),
        "ode_impact_speed": ode_impact_speed,
        "ode_ms":           ode_ms,
        "ode_xs":           xs,
        "ode_ys":           ys,
    }


# ---------------------------------------------------------------------------
# Trajectory Reconstruction (asymmetric power spline, no ODE)
# ---------------------------------------------------------------------------
def _reconstruct_path(R: float, H: float, apex_x: float, n_pts: int = 80):
    """
    Reconstruct 2D drag trajectory from (range, max_altitude, apex_x)
    using a two-piece asymmetric power spline.

    Ascent  (0 → apex_x):     y = H * (1 - ((apex_x - x) / apex_x)^p_asc)
    Descent (apex_x → R):     y = H * (1 - ((x - apex_x) / (R - apex_x))^p_desc)
    """
    apex_x = min(max(apex_x, 0.01), R - 0.01)
    ratio  = apex_x / R

    p_asc  = float(np.clip(1.0 / ratio, 1.35, 2.0))
    p_desc = float(np.clip(0.85 / (1.0 - ratio), 2.0, 3.0))

    # Ascent segment
    xs_asc  = np.linspace(0.0, apex_x, n_pts // 2)
    ys_asc  = H * (1.0 - ((apex_x - xs_asc) / apex_x) ** p_asc)

    # Descent segment
    xs_desc = np.linspace(apex_x, R, n_pts - n_pts // 2)
    ys_desc = H * (1.0 - ((xs_desc - apex_x) / (R - apex_x)) ** p_desc)

    xs = np.concatenate([xs_asc, xs_desc[1:]])
    ys = np.concatenate([ys_asc, ys_desc[1:]])
    ys = np.clip(ys, 0.0, None)

    return xs.tolist(), ys.tolist()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Flight Trajectory Predictor", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    _load_or_train()


# Serve index.html at root
@app.get("/", include_in_schema=False)
async def serve_index():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(str(index), media_type="text/html")


@app.get("/api/metrics")
async def get_metrics():
    return JSONResponse(_metrics)


# ---------------------------------------------------------------------------
# Prediction Request / Response Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    velocity:              float = Field(..., ge=1.0,  le=2000.0, description="m/s")
    angle:                 float = Field(..., ge=1.0,  le=89.0,   description="degrees")
    mass:                  float = Field(..., ge=0.01, le=1000.0, description="kg")
    model_key:             str   = Field("physics_poly")
    include_ode_ground_truth: bool = Field(False)


class PredictResponse(BaseModel):
    model_key:     str
    inference_ms:  float
    speedup_label: str

    # Vacuum analytical bounds
    vacuum_range:    float
    vacuum_altitude: float

    # ML predictions (clamped)
    pred_range:        float
    pred_altitude:     float
    pred_flight_time:  float
    pred_apex_x:       float
    pred_impact_speed: float
    drag_loss_pct:     float
    ke_retained_pct:   float
    asymmetry_ratio:   float

    # Reconstructed spline trajectory
    traj_xs: list
    traj_ys: list

    # Optional ODE ground truth
    ode_range:        Optional[float] = None
    ode_altitude:     Optional[float] = None
    ode_impact_speed: Optional[float] = None
    ode_ms:           Optional[float] = None
    ode_xs:           Optional[list]  = None
    ode_ys:           Optional[list]  = None
    error_range_m:    Optional[float] = None
    error_alt_m:      Optional[float] = None


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if req.model_key not in _models:
        raise HTTPException(
            400,
            f"Unknown model '{req.model_key}'. "
            f"Available: {list(_models.keys())}"
        )

    model = _models[req.model_key]
    v0, ang, m = req.velocity, req.angle, req.mass
    theta = math.radians(ang)

    # --- Vacuum analytical bounds ---
    R_vac = (v0 ** 2) * math.sin(2 * theta) / G
    H_vac = ((v0 * math.sin(theta)) ** 2) / (2 * G)
    R_vac = max(R_vac, 0.05)
    H_vac = max(H_vac, 0.05)

    # --- ML inference ---
    X_in = np.array([[v0, ang, m]], dtype=float)
    t0   = time.perf_counter()
    raw  = model.predict(X_in)[0]
    inf_ms = (time.perf_counter() - t0) * 1000.0

    # raw: [range, max_altitude, flight_time, apex_x, impact_speed]
    pred_range  = float(np.clip(raw[0], 0.05, R_vac))
    pred_alt    = float(np.clip(raw[1], 0.05, H_vac))
    pred_time   = float(max(raw[2], 0.001))
    pred_apex_x = float(np.clip(raw[3], 0.01, pred_range - 0.01))
    pred_impact = float(max(raw[4], 0.01))

    # --- Derived metrics ---
    drag_loss_pct    = float((1.0 - pred_range / R_vac) * 100.0)
    v_impact_no_drag = math.sqrt(v0 ** 2 + 2 * G * 0)  # same speed for vacuum flat
    ke_retained_pct  = float((pred_impact / v0) ** 2 * 100.0)
    asymmetry_ratio  = float(pred_apex_x / pred_range) if pred_range > 0 else 0.5

    # --- Average ODE time for speedup label ---
    ode_avg_ms = _metrics.get("ode_avg_ms", 50.0)
    speedup    = max(ode_avg_ms / max(inf_ms, 0.001), 1.0)
    speedup_label = f"{speedup:.0f}x Faster than ODE"

    # --- Reconstruct spline trajectory ---
    traj_xs, traj_ys = _reconstruct_path(pred_range, pred_alt, pred_apex_x)

    response = {
        "model_key":     req.model_key,
        "inference_ms":  round(inf_ms, 4),
        "speedup_label": speedup_label,
        "vacuum_range":    round(R_vac, 2),
        "vacuum_altitude": round(H_vac, 2),
        "pred_range":        round(pred_range,  2),
        "pred_altitude":     round(pred_alt,     2),
        "pred_flight_time":  round(pred_time,    3),
        "pred_apex_x":       round(pred_apex_x,  2),
        "pred_impact_speed": round(pred_impact,  2),
        "drag_loss_pct":     round(drag_loss_pct, 1),
        "ke_retained_pct":   round(ke_retained_pct, 1),
        "asymmetry_ratio":   round(asymmetry_ratio, 3),
        "traj_xs": [round(x, 2) for x in traj_xs],
        "traj_ys": [round(y, 3) for y in traj_ys],
    }

    # --- Optional ODE ground truth ---
    if req.include_ode_ground_truth:
        ode = _run_ode(v0, ang, m)
        response.update({
            "ode_range":        round(ode["ode_range"], 2),
            "ode_altitude":     round(ode["ode_altitude"], 2),
            "ode_impact_speed": round(ode["ode_impact_speed"], 2),
            "ode_ms":           round(ode["ode_ms"], 2),
            "ode_xs":           [round(x, 2) for x in ode["ode_xs"]],
            "ode_ys":           [round(y, 3) for y in ode["ode_ys"]],
            "error_range_m":    round(abs(pred_range - ode["ode_range"]), 2),
            "error_alt_m":      round(abs(pred_alt   - ode["ode_altitude"]), 2),
        })

    return response


# ---------------------------------------------------------------------------
# Development runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
