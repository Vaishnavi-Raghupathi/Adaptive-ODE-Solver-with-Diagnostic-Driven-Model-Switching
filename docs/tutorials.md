# Three runnable tutorials

From the repository root, install the library with `python -m pip install .`. All three tutorials work with core NumPy/SciPy dependencies. Each output directory must be new. Scripts are included in the source distribution/repository; the installed CLI provides equivalent built-in and CSV entry points without requiring these scripts.

## 1. Equations and parameter error

```bash
python examples/01_known_equations.py --output runs/equations
```

This uses the existing misspecified oscillator. Classical integration of wrong equations cannot fix the parameters. The calibration candidate fits the training observations, validation chooses the method, and the final future block is reported separately. Inspect metrics.csv for the distinction between validation and test scores.

## 2. Missing physics and transfer

```bash
python examples/02_missing_physics.py --output runs/missing-physics
```

The true synthetic oscillator has a cubic restoring force that the assumed linear model omits. The polynomial residual library includes that term. Compare the calibrated linear model and corrected model, then evaluate the fitted correction at a new initial state without retraining. This explicitly favorable polynomial baseline is a test of structured model correction, not proof that arbitrary missing physics can be discovered.

To explore neural comparisons, install `[ml]` and use factories in `adaptive_ode.ml_methods`. Read the saved Days 6–7 report before assuming a neural method will improve the result.

## 3. Observations without supplied equations

```bash
python examples/03_time_series.py --output runs/time-series
python examples/03_time_series.py --csv path/to/your.csv --output runs/my-data
```

The included CSV is explicitly synthetic: a clean oscillator trajectory with time, position and velocity columns. The script compares extrapolation and a learned polynomial ODE using the first time block for fitting. Uploaded target values may contain noise; their held-out errors are not automatically physical ground-truth errors.

Equivalent installed CLI:

```bash
adaptive-solver data examples/data/oscillator.csv --time-column time --states position velocity --output runs/csv-cli
```

## Demo

```bash
python -m pip install '.[demo]'
adaptive-solver demo
# Or from the repository:
python -m streamlit run app.py
```

The demo uses the public API and displays validation selection separately from future test metrics. It supports built-ins, custom equations and numeric CSV data, plotting, downloads and visible failures. The original exploratory UI remains in legacy_app.py and requires requirements-legacy.txt; it retains the audit limitations and is not the validated API.
