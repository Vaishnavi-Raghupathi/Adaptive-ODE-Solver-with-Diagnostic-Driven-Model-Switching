# Adaptive ODE Solver with Diagnostic-Driven Model Switching

An adaptive modeling pipeline for ODE trajectories that:

- solves with a classical ODE model,
- diagnoses residual behavior,
- decides whether to keep the classical model or switch to a Neural ODE,
- reports metrics and visual comparisons.

## Features

- Lorenz-system trajectory generation (`clean`, `noisy`, and `mismatch` scenarios)
- Classical solver baseline using SciPy
- Diagnostic tests for residual analysis
- Rule-based model switching (`classical_ok` vs `neural_ode`)
- Neural ODE training with normalized targets
- Streamlit demo app for interactive runs

## Project Structure

```text
adaptive_ode/
  decision/
  diagnostics/
  evaluation/
  pipeline/
  solvers/
  utils/
app.py
requirements.txt
```

## Setup (Local)

```bash
cd "/Users/vaishnaviraghupathi/Desktop/Adaptive ODE Solver with Diagnostic-Driven Model Switching/Adaptive-ODE-Solver-with-Diagnostic-Driven-Model-Switching"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Pipeline (CLI)

```bash
python -m adaptive_ode.pipeline.run_pipeline
```

## Run the Streamlit App (Local)

```bash
python -m streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## Streamlit App Scenarios

- **Clean Data** → `noise_std=0.0`, `mismatch=False`
- **Noisy Data** → `noise_std=0.5`, `mismatch=False`
- **Model Mismatch** → `noise_std=0.0`, `mismatch=True`

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Create a new app and select this repository + branch `main`.
4. Set main file path to `app.py`.
5. Deploy.

## Notes

- Neural ODE training can be compute-heavy depending on scenario.
- If deployment logs show missing dependencies, verify `requirements.txt` is up to date.
