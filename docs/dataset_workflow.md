# Find a method for your dataset

Adaptive Solver Bench compares candidates on a chronological forecast task. The deliverable is a recommendation report and reproducible metrics, not a claim that a dataset has a universally best solver.

## Start the platform

```bash
python -m pip install '.[demo]'
adaptive-solver demo
```

The default CSV workflow includes an example oscillator that runs without an upload. For your own data, select one numeric time column and the state columns in their intended order. Optionally supply equations using the JSON editor. The interface shows variable-to-column mappings, predictions, validation selection, final test performance, data warnings, failures and runtime trade-offs. Download the recommendation, metrics, trajectories and JSON metadata.

Changing data, equations or benchmark settings hides old results. Changes to the displayed state dimension do not rerun the benchmark.

## What kind of data fits?

- One trajectory with at least 12 rows, finite numeric states and strictly increasing, unique time values. Individual candidates can require more training samples.
- For ODE identification, columns should represent a sufficient observed state. A single measured output is not automatically a complete physical state.
- Explicitly exclude labels, experiment IDs, timestamps encoded as categories and external control inputs from state selection. Inputs-driven identification is not implemented by this workflow. Known time-dependent forcing can be supplied in trusted equations.
- Keep separate trajectories and machines separate. Concatenating independent recordings creates a false physical trajectory.
- Use consistent units. Irregular sampling is accepted, but derivative fitting may be unreliable, particularly with noisy or tightly spaced measurements.

No missing-value filling, sorting, duplicate removal or resampling is performed silently. Correct invalid data explicitly. The platform does not infer stiffness, chaos, measurement-noise level or physical observability from summary statistics.

## Two comparison modes

**Without equations:** persistence (last training observation), a least-squares linear time trend, cubic spline extrapolation, a learned linear ODE and a learned quadratic ODE. An optional MLP predicts states from time. These are forecasting comparisons. The spline can extrapolate badly and learned derivatives can amplify noise. Default learned ODE fits have a 256-feature budget; larger models fail explicitly rather than allocating an unbounded feature library.

**With equations:** the same candidates plus RK4, RK45, DOP853, BDF, Radau and LSODA. Equations must describe every supplied state, in column order and matching units. Integrators use the first measured state as their initial condition, without true-state resets later. Supplied parameters are fixed; generic physical parameter calibration is not implemented. Observed-data error combines equation mismatch, initial-state error, noise and integration error; it does not isolate numerical accuracy. Default solver settings are tested, not all possible tolerance/step-size configurations.

## Python

```python
from adaptive_ode import BenchmarkConfig, benchmark_data

config = BenchmarkConfig(selection_metric="normalized_rmse")
result = benchmark_data(times, states, config=config)
# Or add known equations:
# result = benchmark_data(times, states, rhs=lambda t, y: -y, config=config)

print(result.selected_method)
print(result.assessment)
print(result.report())
result.save("runs/dataset-comparison")
```

Supply explicit `Candidate` objects for additional research methods. Use either `rhs` for the default integrator suite or explicit `candidates`, not both. `Candidate.fit` receives training samples only. The platform cannot prevent a custom callback from capturing future labels from external state.

## CLI and equation files

```bash
adaptive-solver data examples/data/oscillator.csv \
  --time-column time --states position velocity \
  --metric normalized_rmse --output runs/observations

adaptive-solver data examples/data/oscillator.csv \
  --time-column time --states position velocity \
  --equations examples/data/oscillator_equations.json \
  --metric normalized_rmse --output runs/with-equations
```

Equation files require the `custom` extra (already included in `demo`). The oscillator specification is:

```json
{
  "name": "Damped oscillator",
  "variables": ["position", "velocity"],
  "parameters": {"c": 0.1, "k": 4},
  "derivatives": ["velocity", "-c*velocity-k*position"]
}
```

Expressions pass through the existing restricted arithmetic parser, not Python `eval`. Variable names are aliases in selected column order. This is not a facility for running arbitrary uploaded Python.

## How selection works

The initial block fits each model, the next block selects variants and the method, and the final future block reports performance. Defaults are 60%/20%/20%. There is no refit after selection. Candidates must return a finite complete prediction across the grid; a future numerical failure makes a candidate ineligible even though future state labels are not inspected for selection.

The UI defaults to `normalized_rmse`; Python/CLI retain `mse` for compatibility. Normalized RMSE is:

```text
sqrt(mean(((observation - prediction) / per_state_training_scale) ** 2))
```

Scale is the population standard deviation of retained training values. Numerically constant states use their maximum absolute training magnitude, or one if identically zero. This makes nonconstant dimensions comparable across units. Zero-constant states have no data-derived unit scale, so their normalized score still depends on the chosen units. This does not make a fitted method itself invariant to rescaling or establish physical importance weights.

The profile uses retained training observations, never future state values. Raw MSE/RMSE remain available. Final test scores use observed targets for uploaded data, which may contain noise. Test scores do not reselect the winner, including when final scoring fails.

The accuracy/runtime Pareto set contains successful methods for which another method is not both at least as accurate on validation and at least as fast under the chosen runtime cost, with a strict improvement in one. It describes trade-offs. The default policy minimizes error; `fastest_within_target` instead chooses the lowest measured cost among configurations with validation score at most the specified target. No qualifying method yields no recommendation. Prediction cost is a median over configured repeats; fit-plus-prediction adds one fit. Search time includes all variants and repeated prediction work. Local timings and a validation threshold do not guarantee runtime or accuracy on future data.

Use `--policy fastest_within_target --accuracy-target 0.05 --timing-repeats 3 --solver-sweep` to explore this policy in the CLI. Adaptive solvers are tried at rtol 1e-3, 1e-5, 1e-7 (atol=rtol*1e-2); RK4 uses three maximum step sizes derived from training sampling. All variant scores and chosen settings are exported. See the independent-experiment protocol for exact settings and interpretation.

## Saved deliverables and current scope

`save()` creates a new directory with `report.md`, `results.json` (schema 2), `metrics.csv` and numeric `predictions.npz`. Metadata includes configuration, split indices, training scales, candidate diagnostics and environment versions. UI/CLI context includes the equation specification when supplied as JSON and a dataset source identifier/hash. Python RHS callbacks and fitted models are not serialized; retain their source to reproduce custom models. Existing output directories are never overwritten.

For independent training/validation/test trajectories and conditional test-trajectory bootstrap intervals, use the separate [independent-experiment workflow](independent_experiments.md). Rolling-origin evaluation, general external-input learning, arbitrary neural ODE/PINN training, hosted accounts and public PyPI publication remain outside this release. Built-in demonstrations are synthetic. A recommendation for uploaded data requires the researcher's own independent validation before broader conclusions.
