# Adaptive ODE Solver with Diagnostic-Driven Model Switching

This project models Lorenz system trajectories with a classical ODE solver, analyzes residuals with statistical diagnostics, and automatically switches to a Neural ODE when diagnostics indicate the classical model is no longer sufficient. It supports three scenarios: Clean Data, Noisy Data, and Model Mismatch.

## How It Works

1. Generate Lorenz trajectory data.
2. Run the classical solver.
3. Diagnose residuals.
4. Decide whether classical is enough or Neural ODE is needed.
5. Train Neural ODE if needed.
6. Report metrics and plots.

## Results

| Scenario | Decision | Classical MSE | Neural MSE | Improvement |
|---|---|---|---|---|
| Clean Data | classical_ok | 0.017 | N/A | N/A |
| Noisy Data | classical_ok | 1.39 | N/A | N/A |
| Model Mismatch | neural_ode | 1.81 | 0.24 | +86.5% |

## Tech Stack

Python: core language and project structure.

PyTorch + torchdiffeq (Neural ODE): neural dynamics modeling and ODE integration.

SciPy (classical solver + statistical tests): classical ODE solving and numerical routines.

Statsmodels (Breusch-Pagan, Ljung-Box, ADF): residual diagnostic testing.

Streamlit (demo UI): interactive scenario selection and results visualization.

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Select this repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

Streamlit Cloud will install dependencies from `requirements.txt` automatically.
