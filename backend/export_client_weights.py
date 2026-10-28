"""
backend/export_client_weights.py
================================
Exports trained model weights and benchmark metrics to frontend/model_weights.json
so the entire web app can run 100% client-side on GitHub Pages without any backend.
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import joblib
import numpy as np
MODELS_PATH = PROJECT_DIR / "backend" / "models" / "trajectory_models.joblib"
METRICS_PATH = PROJECT_DIR / "backend" / "models" / "metrics.json"
OUT_PATH = PROJECT_DIR / "frontend" / "model_weights.json"

models = joblib.load(MODELS_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)

poly_pipe = models["physics_poly"]
scaler = poly_pipe.named_steps["scaler"]
poly = poly_pipe.named_steps["poly"]
ridge = poly_pipe.named_steps["ridge"]

lin = models["simple_linear"]

# Export Decision Tree structure
dt = models["decision_tree"].tree_
dt_dict = {
    "node_count": int(dt.node_count),
    "children_left": dt.children_left.tolist(),
    "children_right": dt.children_right.tolist(),
    "feature": dt.feature.tolist(),
    "threshold": dt.threshold.tolist(),
    "value": dt.value.squeeze(axis=-1).tolist()
}

export_data = {
    "metrics": metrics,
    "physics_poly": {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "poly_powers": poly.powers_.tolist(),
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": ridge.intercept_.tolist(),
    },
    "simple_linear": {
        "coef": lin.coef_.tolist(),
        "intercept": lin.intercept_.tolist(),
    },
    "decision_tree": dt_dict
}

with open(OUT_PATH, "w") as f:
    json.dump(export_data, f)

print(f"Exported client weights: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
