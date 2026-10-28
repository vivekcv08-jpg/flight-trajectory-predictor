"""
backend/train_model.py
======================
Flight Trajectory ML Training Pipeline

Physics: 2D point-mass with quadratic aerodynamic drag
  dx/dt  = vx
  dy/dt  = vy
  dvx/dt = -(k_drag/m) * v * vx
  dvy/dt = -g - (k_drag/m) * v * vy
  where v = hypot(vx, vy)

Trains 4 scikit-learn models, benchmarks, and exports:
  backend/models/trajectory_models.joblib
  backend/models/metrics.json
"""

import os
import sys
import json
import time
import warnings
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from scipy.integrate import solve_ivp

# Shared physics constants and feature expansion (importable for joblib)
from backend.physics import G, K_DRAG, physics_feature_expansion

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = SCRIPT_DIR / "models"
DATASET_CSV = DATA_DIR / "flight_dataset.csv"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# ODE Physics (used for dataset generation and ground-truth comparisons)
# ---------------------------------------------------------------------------
def _eom(t, state, m):
    """Equations of motion for 2D drag projectile."""
    x, y, vx, vy = state
    v = math.hypot(vx, vy)
    k = K_DRAG / m
    return [vx, vy, -k * v * vx, -G - k * v * vy]


def simulate_trajectory(v0: float, angle_deg: float, m: float,
                        rtol: float = 1e-6, atol: float = 1e-8):
    """
    Simulate one trajectory using solve_ivp RK45.
    Returns dict with: range, max_altitude, flight_time, apex_x, impact_speed,
    and solve_time_ms.
    Also returns full (xs, ys) arrays for plotting.
    """
    theta = math.radians(angle_deg)
    vx0   = v0 * math.cos(theta)
    vy0   = v0 * math.sin(theta)
    state0 = [0.0, 0.0, vx0, vy0]

    t_start = time.perf_counter()

    # Event 1: apex (vy = 0, descending)
    def apex_event(t, s, m):
        return s[3]
    apex_event.terminal  = False
    apex_event.direction = -1

    # Event 2: ground impact (y = 0, descending)
    def ground_event(t, s, m):
        return s[1]
    ground_event.terminal  = True
    ground_event.direction = -1

    sol = solve_ivp(
        _eom, [0, 2000], state0,
        args=(m,),
        method='RK45',
        rtol=rtol, atol=atol,
        events=[apex_event, ground_event],
        dense_output=False,
        max_step=1.0,
    )

    t_end = time.perf_counter()
    solve_ms = (t_end - t_start) * 1000.0

    # Extract results
    xs = sol.y[0]
    ys = sol.y[1]

    # Ground impact
    if sol.t_events[1].size > 0:
        te = sol.t_events[1][-1]
        ye = sol.y_events[1][-1]
        landing_x    = float(ye[0])
        flight_time  = float(te)
        impact_speed = float(math.hypot(ye[2], ye[3]))
    else:
        landing_x    = float(xs[-1])
        flight_time  = float(sol.t[-1])
        impact_speed = float(math.hypot(sol.y[2, -1], sol.y[3, -1]))

    # Apex
    if sol.t_events[0].size > 0:
        ye_apex = sol.y_events[0][-1]
        apex_x   = float(ye_apex[0])
        max_alt  = float(ye_apex[1])
    else:
        idx_max = int(np.argmax(ys))
        apex_x  = float(xs[idx_max])
        max_alt = float(ys[idx_max])

    return {
        "range":        max(landing_x, 0.05),
        "max_altitude": max(max_alt, 0.05),
        "flight_time":  max(flight_time, 0.001),
        "apex_x":       max(apex_x, 0.01),
        "impact_speed": max(impact_speed, 0.01),
        "solve_time_ms": solve_ms,
        "xs": xs.tolist(),
        "ys": ys.tolist(),
    }


# ---------------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------------
def generate_dataset() -> pd.DataFrame:
    """Generate 5,000-row dataset via solve_ivp if CSV missing."""
    velocities  = np.linspace(20, 300, 20)
    angles_deg  = np.linspace(5, 85, 25)
    masses      = np.logspace(np.log10(0.5), np.log10(50), 10)

    rows = []
    total = len(velocities) * len(angles_deg) * len(masses)
    count = 0

    print(f"Generating {total} trajectories via Python solve_ivp...")
    for v0 in velocities:
        for ang in angles_deg:
            for m in masses:
                result = simulate_trajectory(v0, ang, m)
                rows.append({
                    "velocity":      v0,
                    "angle":         ang,
                    "mass":          m,
                    "range":         result["range"],
                    "max_altitude":  result["max_altitude"],
                    "flight_time":   result["flight_time"],
                    "apex_x":        result["apex_x"],
                    "impact_speed":  result["impact_speed"],
                })
                count += 1
                if count % 500 == 0:
                    print(f"  {count}/{total} complete...")

    df = pd.DataFrame(rows)
    df.to_csv(DATASET_CSV, index=False)
    print(f"Dataset saved: {DATASET_CSV}  ({len(df)} rows)")
    return df


# physics_feature_expansion imported from backend.physics


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
def build_models() -> dict:
    return {
        "physics_poly": Pipeline([
            ("physics_expand", FunctionTransformer(
                physics_feature_expansion, validate=False)),
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=3, include_bias=True)),
            ("ridge",  Ridge(alpha=0.01)),
        ]),
        "decision_tree": DecisionTreeRegressor(max_depth=14, random_state=42),
        "simple_linear": LinearRegression(),
        "extra_trees":   ExtraTreesRegressor(
            n_estimators=50, random_state=42, n_jobs=1),
    }


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_and_export():
    """Main pipeline: load/generate data → train → evaluate → export."""

    # --- Load or generate dataset ---
    if DATASET_CSV.exists():
        print(f"Loading existing dataset: {DATASET_CSV}")
        df = pd.read_csv(DATASET_CSV)
    else:
        df = generate_dataset()

    print(f"Dataset shape: {df.shape}")

    # --- Feature / target split ---
    feature_cols = ["velocity", "angle", "mass"]
    target_cols  = ["range", "max_altitude", "flight_time", "apex_x", "impact_speed"]

    X = df[feature_cols].values
    y = df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # --- Benchmark ODE solve time (average over 10 samples) ---
    print("\nBenchmarking ODE solve time...")
    ode_times = []
    for i in range(min(10, len(X_test))):
        row = X_test[i]
        res = simulate_trajectory(row[0], row[1], row[2])
        ode_times.append(res["solve_time_ms"])
    avg_ode_ms = float(np.mean(ode_times))
    print(f"  Average ODE solve time: {avg_ode_ms:.2f} ms")

    # --- Train & evaluate all models ---
    models_dict = build_models()
    trained     = {}
    metrics     = {"ode_avg_ms": avg_ode_ms, "models": {}}

    for key, model in models_dict.items():
        print(f"\nTraining [{key}]...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_ms = (time.perf_counter() - t0) * 1000

        # Evaluate on test set
        y_pred = model.predict(X_test)

        r2_range   = float(r2_score(y_test[:, 0], y_pred[:, 0]))
        r2_alt     = float(r2_score(y_test[:, 1], y_pred[:, 1]))
        mae_range  = float(mean_absolute_error(y_test[:, 0], y_pred[:, 0]))
        mae_alt    = float(mean_absolute_error(y_test[:, 1], y_pred[:, 1]))

        # Single-sample inference time (average over 50 runs)
        single_X = X_test[:1]
        inf_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model.predict(single_X)
            inf_times.append((time.perf_counter() - t0) * 1000)
        inf_ms = float(np.median(inf_times))
        speedup = float(avg_ode_ms / max(inf_ms, 1e-6))

        print(f"  R²_range={r2_range:.4f}  R²_alt={r2_alt:.4f}  "
              f"MAE_range={mae_range:.1f}m  MAE_alt={mae_alt:.1f}m  "
              f"inf={inf_ms:.3f}ms  speedup={speedup:.1f}x")

        trained[key] = model
        metrics["models"][key] = {
            "r2_range":       r2_range,
            "r2_altitude":    r2_alt,
            "mae_range_m":    mae_range,
            "mae_altitude_m": mae_alt,
            "inference_ms":   inf_ms,
            "speedup_vs_ode": speedup,
            "train_ms":       train_ms,
        }

    # --- Serialize ---
    joblib_path  = MODELS_DIR / "trajectory_models.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    joblib.dump(trained, joblib_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved: {joblib_path}")
    print(f"Metrics saved: {metrics_path}")
    return trained, metrics


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_export()
