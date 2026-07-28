# Adaptive Solver Selection for Dynamical Systems

This project benchmarks multiple solver strategies for a selected dynamical system and recommends the best method from held-out trajectory performance. It is designed as a research-assistant style web app: instead of manually trying classical, hybrid, and physics-informed approaches, a researcher can run a small solver tournament and inspect the tradeoffs.

## Solver Families

- Classical explicit solvers: fixed-step RK4, RK45, DOP853.
- Classical implicit solvers: BDF, Radau, LSODA.
- Hybrid residual correction: RK4 baseline plus a neural network trained to predict the residual.
- PINN-style surrogate: neural trajectory model trained with data loss and an ODE residual penalty.

## Built-In Systems

- Lorenz attractor: nonlinear chaotic dynamics.
- Van der Pol oscillator: nonlinear oscillator with moderate stiffness.
- Robertson chemical kinetics: highly stiff reaction dynamics.
- Misspecified damped oscillator: true dynamics differ from the classical baseline, making residual correction useful.

## How It Works

1. Generate a high-accuracy reference trajectory for the chosen system.
2. Add optional observation noise to create training observations.
3. Split the trajectory into train and held-out regions.
4. Run all enabled solver candidates.
5. Score candidates on held-out MSE, RMSE, MAE, and runtime.
6. Recommend the solver with the lowest held-out error.
7. Show trajectory, residual, diagnostics, and ranking plots.

## Uploaded Data Mode

Researchers can also upload a CSV when the governing equations are unknown. The app expects one numeric time column and one or more numeric state columns.

For uploaded data, the app runs data-driven candidates:

- Cubic spline trajectory fit.
- MLP trajectory surrogate.
- Polynomial learned ODE, a lightweight SINDy-style dynamics model.

Uploaded data is split chronologically: the first segment trains the data-driven models and the later segment is held out for evaluation.

## Custom Equation Mode

Researchers can enter their own dynamical system by specifying state variables, parameters, derivative equations, initial conditions, and time span. Expressions are parsed with a restricted SymPy parser and converted into a numerical right-hand side for the same solver-selection benchmark.

Example:

```text
Variables: x, y, z
Parameters: sigma=10, rho=28, beta=8/3
Equations:
sigma*(y - x)
x*(rho - z) - y
x*y - beta*z
Initial condition: 1, 1, 1
```

## Interactive Results

The app displays editable Plotly charts with selectable state dimensions, solver curves, residual views, log-scale ranking, hover inspection, and downloadable metrics CSV, trajectory CSV, and HTML plots.

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publish

This app is ready to publish on Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Create a new Streamlit app.
3. Select this repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

For a public demo, leave the PINN option off by default because it can be slower on free cloud resources.

## Notes

The PINN candidate is intentionally lightweight so it can run inside a demo app. For serious stiff-system research, implicit classical methods such as BDF, Radau, or LSODA should remain strong baselines, and the app allows them to win when their held-out metrics are best.
