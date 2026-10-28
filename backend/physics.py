"""
backend/physics.py
==================
Shared physics constants and feature expansion function.
Must be importable from BOTH train_model.py and main.py
so that joblib can unpickle the FunctionTransformer correctly.
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Physics Constants
# ---------------------------------------------------------------------------
G       = 9.81          # m/s²
RHO0    = 1.225         # kg/m³
CD      = 0.30
A_AREA  = 0.02          # m²
K_DRAG  = 0.5 * RHO0 * CD * A_AREA   # = 0.003675  kg/m


# ---------------------------------------------------------------------------
# Physics-informed Feature Expansion
# Must live here (not in __main__) so joblib can unpickle it.
# ---------------------------------------------------------------------------
def physics_feature_expansion(X: np.ndarray) -> np.ndarray:
    """
    Input columns: [velocity, angle_deg, mass]
    Output 10 columns:
      [v, angle_rad, m, log(m), 1/m,
       sin(angle), cos(angle), sin(2*angle),
       v*cos(angle), v*sin(angle)]
    """
    v   = X[:, 0]
    ang = X[:, 1]
    m   = X[:, 2]
    angle_rad = np.radians(ang)
    return np.column_stack([
        v,
        angle_rad,
        m,
        np.log(np.maximum(m, 1e-9)),
        1.0 / np.maximum(m, 1e-9),
        np.sin(angle_rad),
        np.cos(angle_rad),
        np.sin(2.0 * angle_rad),
        v * np.cos(angle_rad),
        v * np.sin(angle_rad),
    ])
