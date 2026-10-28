# 🚀 Instant Flight Trajectory Predictor

An aerospace telemetry dashboard powered by machine learning that replaces numerical differential equation integration (ODE) with sub-millisecond scikit-learn models for 2D point-mass projectile flight dynamics with quadratic aerodynamic drag.

---

## 🌟 Live Demo on GitHub Pages
👉 **Live Web App**: [https://your-username.github.io/flight-trajectory-predictor/](https://your-username.github.io/flight-trajectory-predictor/) *(replace with your GitHub username)*

The web app runs **100% in-browser on GitHub Pages** using client-side polynomial physics expansion and Runge-Kutta numerical verification with zero server requirements!

---

## ⚡ Physics Specifications

Point-mass projectile subjected to Earth gravity ($g = 9.81\text{ m/s}^2$) and quadratic aerodynamic drag:

$$\begin{aligned}
\frac{dx}{dt} &= v_x, \quad \frac{dy}{dt} = v_y \\
\frac{dv_x}{dt} &= -\frac{k_{\text{drag}}}{m} v v_x, \quad \frac{dv_y}{dt} = -g - \frac{k_{\text{drag}}}{m} v v_y \\
v &= \sqrt{v_x^2 + v_y^2}, \quad k_{\text{drag}} = \frac{1}{2}\rho_0 C_d A = 0.003675\text{ kg/m}
\end{aligned}$$

- Air density: $\rho_0 = 1.225\text{ kg/m}^3$
- Drag coefficient: $C_d = 0.30$
- Cross-sectional area: $A = 0.02\text{ m}^2$

---

## 📊 Benchmark Metrics (Holdout Test Set)

| Model Key | Algorithm | $R^2$ Range | $R^2$ Alt | MAE Range (m) | MAE Alt (m) | Single Inf. Latency | Speedup vs ODE |
|---|---|---|---|---|---|---|---|
| **`physics_poly`** *(Default)* | Physics Features + Poly(3) + Ridge | **0.9991** | **0.9995** | **22.3 m** | **7.4 m** | **0.266 ms** | **3.9x** |
| **`extra_trees`** | ExtraTreesRegressor ($N=50$) | **0.9992** | **0.9985** | **14.4 m** | **8.6 m** | 1.851 ms | 0.6x |
| **`decision_tree`** | DecisionTreeRegressor ($D=14$) | 0.9935 | 0.9865 | 47.1 m | 32.5 m | **0.056 ms** | **18.6x** |
| **`simple_linear`** *(Baseline)* | Naive LinearRegression | 0.5358 | 0.5960 | 511.0 m | 232.7 m | 0.049 ms | 21.2x |

---

## 🛠️ Local Installation & Development

### 1. Clone repository
```bash
git clone https://github.com/your-username/flight-trajectory-predictor.git
cd flight-trajectory-predictor
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate Data & Train Models
```bash
python -m backend.train_model
```

### 4. Launch FastAPI Inference Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
