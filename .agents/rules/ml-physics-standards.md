---
trigger: always_on
---

# ML Physics Standards — Workspace Rule

## Physical Constraint Enforcement

All ML trajectory predictions MUST be clamped within physical bounds before returning to any caller.

### Range Constraints
- **Minimum**: `predicted_range >= 0.05` meters (never negative or zero)
- **Maximum**: `predicted_range <= R_vacuum = v² * sin(2θ) / g` (drag-free upper bound)

### Altitude Constraints
- **Minimum**: `predicted_altitude >= 0.05` meters (never negative or zero)
- **Maximum**: `predicted_altitude <= H_vacuum = (v·sin(θ))² / (2·g)` (drag-free upper bound)

### Apex X Constraints
- **Minimum**: `apex_x >= 0.01` meters
- **Maximum**: `apex_x <= predicted_range - 0.01` (must be strictly inside range)

### Flight Time Constraints
- **Minimum**: `flight_time >= 0.001` seconds

### Impact Speed Constraints
- **Minimum**: `impact_speed >= 0.01` m/s

## Implementation Note
These constraints are enforced in `backend/main.py` at the `/api/predict` endpoint using the analytical vacuum trajectory as the upper bound. The `physics_poly` model (default) is trained with physics-aware feature expansion to naturally respect these bounds, but clamping is applied to all models.

## Constants Reference
```python
g       = 9.81        # m/s²
rho0    = 1.225       # kg/m³
Cd      = 0.30        # dimensionless
A       = 0.02        # m²
k_drag  = 0.003675    # kg/m  (= 0.5 * rho0 * Cd * A)
```
