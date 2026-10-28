# Adaptive Solver Bench

An open-source Python library and Streamlit platform for comparing numerical solvers and forecasting models on dynamical-system time series. Upload a CSV, optionally supply governing equations, compare tested methods, and export a recommendation with validation error, future test error, runtime and failures.

Numerical integrators need equations. Without equations the platform compares forecasting models, including learned ODEs; it cannot identify the best numerical integrator from a CSV alone. Supply one finite numeric trajectory for chronological evaluation, or separate training/validation/test experiments for initial-state transfer. Do not concatenate independent experiments into one physical trajectory.

**Status:** 0.1.0 alpha release candidate. Installable from this repository or a built wheel; not yet published on PyPI. This is a benchmarking and evaluation library, not a new universal solver or a guarantee that neural methods improve numerical integration.

## Install

Python 3.11+ is targeted; local release verification uses Python 3.13 on macOS ARM64. The configured CI matrix also targets Python 3.11–3.13 on Linux; remote runs remain to be performed after pushing.

```bash
python -m pip install .
# Optional features:
python -m pip install '.[ml,plots]'
python -m pip install '.[demo]'
```

The core requires only NumPy and SciPy. Extras separate scikit-learn, plotting, custom equation parsing, the web demo, research tools and development tools. Distribution name: **adaptive-solver-bench**. Import: **adaptive_ode**. Command: **adaptive-solver**.

## Use from Python

```python
from adaptive_ode import BenchmarkConfig, benchmark

result = benchmark("mismatch", config=BenchmarkConfig(num_points=301))
print(result.selected_method)
print(result.ranking)  # ordered by validation MSE, never final test MSE
result.save("runs/oscillator")  # new directory; JSON, CSV, numeric NPZ
```

For observations without supplied equations:

```python
from adaptive_ode import benchmark_data

result = benchmark_data(times, states,
                        config=BenchmarkConfig(selection_metric="normalized_rmse"))
print(result.assessment)
print(result.report())  # human-readable Markdown recommendation
result.save("runs/my-data")
```

If you have equations, compare six numerical integrators against those same measured observations:

```python
def rhs(t, state):
    position, velocity = state
    return [velocity, -0.1 * velocity - 4 * position]

result = benchmark_data(times, states, rhs=rhs,
                        config=BenchmarkConfig(selection_metric="normalized_rmse"))
```

The recommendation is empirical: best among successful tested candidates on your validation interval. It is not a universal solver prediction. See the [dataset workflow](docs/dataset_workflow.md) for CSV requirements, equation files, scoring and limitations.

## Independent experiments and accuracy requirements

Choose **Independent trajectories** in the interface, or use the public API:

```python
from adaptive_ode import Trajectory, benchmark_trajectories

config = BenchmarkConfig(selection_metric="normalized_rmse",
                         selection_policy="fastest_within_target", accuracy_target=0.05,
                         timing_repeats=3, solver_sweep=True)
result = benchmark_trajectories(train=training_experiments, validation=validation_experiments,
                                test=test_experiments, rhs=rhs, config=config)
```

Each experiment is a `Trajectory(times, states, "unique-experiment-name")`. Complete training experiments fit the model; different validation experiments choose the method and settings. Test experiments are predicted only after selection is frozen. Only each new experiment's initial state is supplied. No method satisfying the accuracy target yields **no recommendation**. Runtime selection can use repeated prediction time or fitting plus prediction; it never uses final test error.

The [independent-experiment protocol](docs/independent_experiments.md) documents equal experiment weighting, initial-state transfer, tolerance/step sweeps, test-trajectory bootstrap intervals, manifests and limitations. Run the synthetic nine-experiment example:

```bash
python examples/04_independent_trajectories.py --output runs/independent-example
```

Supply a `DynamicalSystem` for custom Python equations or `Candidate`/`FittedModel` callbacks for your own methods. See the [API reference](docs/api.md).

## CLI and demo

```bash
adaptive-solver benchmark --system mismatch --output runs/first
adaptive-solver data examples/data/oscillator.csv --output runs/uploaded
adaptive-solver demo  # install the demo extra first
# Repository entry point:
python -m streamlit run app.py
```

The demo uses the same public API, with distinct validation and final-test metrics, editable plots, CSV/JSON downloads and visible failures. The original exploratory app is preserved as legacy_app.py with requirements-legacy.txt; it retains the [Day 1 audit limitations](docs/day01_audit.md).

The platform opens in CSV mode, includes runnable single- and independent-experiment examples, and supports optional equations for measured data. It exports a Markdown recommendation, metrics, trajectories and evaluation metadata. Reports include baseline comparisons, all solver-setting trials, failure diagnostics and accuracy/runtime trade-offs. Changing benchmark inputs hides previous recommendations until a new run is completed.

See the [deployment guide](docs/deployment.md) for hosting this Python app on Streamlit Community Cloud. The example oscillator is available immediately on first load; comparisons run when requested.

## What is implemented

- Built-in Lorenz, Van der Pol, Robertson and misspecified oscillator examples, plus custom Python systems.
- RK4, RK45, DOP853, BDF, Radau and LSODA integration; bounded parameter calibration for the oscillator.
- Observations-only persistence, linear trend, spline and linear/quadratic learned-ODE candidates; optional neural trajectory models.
- Oscillator-specific state-based polynomial and neural acceleration corrections; supported fitted dynamics can transfer to new initial states without refitting.
- Validation-tuned variants, optional training-normalized scoring, separate fit/prediction/search costs, residual summaries, failure budgets and reproducible research artifacts.

The supported library API excludes the legacy PINN and PyTorch Neural ODE implementations, which have unresolved evaluation limitations. General dynamical-system identification and sparse SINDy are not claimed.

## What the experiments found

The saved studies support a narrower, useful conclusion: calibrate known model structure first; when physics is missing, compare simple structured corrections before assuming neural networks are necessary.

| Study | Evidence |
|---|---|
| Day 5: parameter mismatch | 25 cases. Calibration selected in all 20 nonzero-mismatch cases; the trajectory residual NN did not beat calibration on future test data. |
| Days 6–7: noise, reduced data and missing cubic force | 60 cases plus three unseen initial states. Calibration selected in 34 cases and polynomial correction in 26; neither neural method was selected. Instability/failures remain visible. |
| Day 8: classical controls | Four built-in systems, three deterministic timing repeats each. Tight numerical references cross-checked; unstable/budget-exhausted candidates recorded. |

These are synthetic, fully observed, low-dimensional studies. The polynomial library intentionally contains the missing cubic term. They establish neither general-purpose superiority nor unrestricted generalization. Read the [Days 1–5 report](docs/day01_day05_summary.md), [robustness report](experiments/results/day06_07_robustness/report.md), [classical report](experiments/results/day08_classical/report.md), and [release assessment](docs/days06_14_summary.md).

## Tutorials, tests and build

Five standalone examples cover known equations, missing physics with transfer, observations-only data, independent synthetic experiments and [measured double-pendulum recordings](docs/experimental_pendulum.md). The measured-data pilot documents a no-recommendation result when the current models miss the validation target. See the [tutorial guide](docs/tutorials.md). Contributions and method adapters are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
python -m pip install -e '.[research,demo,dev]'
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

[CI](.github/workflows/ci.yml) tests and builds distributions; [manual release automation](.github/workflows/release.yml) defaults to artifacts only. Remote CI and PyPI publishing are separate steps described in the [release guide](docs/releasing.md).

## Reproduce research

Use the frozen configs/protocols and exact source/environment snapshots in experiments/results. The original requirements-research.lock belongs to Day 5; later studies save their own environments and requirements-dev.lock snapshot. Saved runs refuse overwrites.

```bash
python -m adaptive_ode.experiments.robustness --config experiments/configs/day06_07_robustness.json --output experiments/results/robustness-reproduction
python -m adaptive_ode.experiments.classical_controls --config experiments/configs/day08_classical.json --output experiments/results/classical-reproduction
```

Research scripts/configs and large saved results are repository artifacts; the wheel contains the reusable library and demo. See the [research guide](experiments/README.md) and written [robustness protocol](docs/robustness_protocol.md).

Apache-2.0 license; see LICENSE.
