# Runnable tutorials

From the repository root, install the library with `python -m pip install .`. All five tutorials work with core NumPy/SciPy dependencies. Each output directory must be new. Scripts are included in the source distribution/repository; the installed CLI provides equivalent built-in, CSV and independent-experiment entry points without requiring these scripts.

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

## 4. Independent experiments and accuracy targets

```bash
python examples/04_independent_trajectories.py --output runs/independent-example
```

Nine synthetic oscillator experiments have separate training/validation/test roles and different initial states. The policy selects the lowest measured prediction cost among configurations meeting a validation accuracy target. Solver tolerances and RK4 step sizes are compared within method groups. The example exports reports, arrays, CSV inputs and a reusable manifest. Read the [independent-experiment protocol](independent_experiments.md) before interpreting bootstrap intervals or timing ranks.

## Demo

```bash
python -m pip install '.[demo]'
adaptive-solver demo
# Or from the repository:
python -m streamlit run app.py
```

The demo uses the public API and displays validation selection separately from future test metrics. It supports built-ins, custom equations and numeric CSV data, plotting, downloads and visible failures. The original exploratory UI remains in legacy_app.py and requires requirements-legacy.txt; it retains the audit limitations and is not the validated API.

For the dataset-first platform, start with the [dataset workflow](dataset_workflow.md). It covers optional equations for measured data, training-normalized scoring, baseline comparisons and downloadable recommendation reports.

## 5. Measured double-pendulum recordings

```bash
python examples/05_experimental_pendulum.py --download \
  --source runs/experimental-pendulum/source --output runs/experimental-pendulum/pilot
```

Downloads four pinned, checksum-verified experimental recordings and upstream attribution (about 19.5 MB). Two recordings train the models, one selects and one tests. This five-second data-driven pilot intentionally allows a no-recommendation outcome. Read the [dataset assessment and initial results](experimental_pendulum.md) for the fixed protocol and scientific limits.
