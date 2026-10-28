# Instant Flight Trajectory Predictor — Agent Context

## Project Overview
A full-stack ML web application that predicts 2D flight trajectories of a point-mass projectile with quadratic aerodynamic drag using trained scikit-learn models, served by a FastAPI backend and interactive Chart.js frontend.

## Server
- **URL**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs

## Directory Structure
```
flight-predictor/
├── AGENTS.md                        ← This file
├── .agents/rules/ml-physics-standards.md
├── requirements.txt
├── data/
│   └── flight_dataset.csv           ← 5,000-row generated dataset
├── matlab/
│   └── generate_flight_data.m       ← MATLAB ODE45 data generator
├── backend/
│   ├── train_model.py               ← Dataset generation + ML training
│   ├── main.py                      ← FastAPI server
│   └── models/
│       ├── trajectory_models.joblib ← Serialized trained models
│       └── metrics.json             ← Benchmark metrics
└── frontend/
    └── index.html                   ← Dark-mode aerospace dashboard
```

## Physics Constants
| Constant | Value | Units |
|---|---|---|
| g (gravity) | 9.81 | m/s² |
| ρ₀ (air density) | 1.225 | kg/m³ |
| Cd (drag coefficient) | 0.30 | — |
| A (cross-sectional area) | 0.02 | m² |
| k_drag = 0.5·ρ₀·Cd·A | 0.003675 | kg/m |

## Equations of Motion (2D Point-Mass with Quadratic Drag)
```
dx/dt  = vx
dy/dt  = vy
dvx/dt = -(k_drag / m) * sqrt(vx²+vy²) * vx
dvy/dt = -g - (k_drag / m) * sqrt(vx²+vy²) * vy
```

## ML Models
| Key | Algorithm | Notes |
|---|---|---|
| physics_poly | Physics-informed Ridge Regression | Default; physics-expanded features + PolyFeatures(3) |
| decision_tree | DecisionTreeRegressor(max_depth=14) | Fast, nonlinear |
| extra_trees | ExtraTreesRegressor(n_estimators=50) | Ensemble |
| simple_linear | LinearRegression | Baseline/demo of underfitting |

## API Endpoints
- `GET  /`                → Serves frontend/index.html
- `GET  /api/metrics`     → Returns benchmark metrics JSON
- `POST /api/predict`     → Runs ML prediction, returns trajectory + optionally ODE ground truth
