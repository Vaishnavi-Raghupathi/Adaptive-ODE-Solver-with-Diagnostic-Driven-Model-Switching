# Public API, version 0.1

Install the repository using `python -m pip install .`; the distribution is named **adaptive-solver-bench**, the import package **adaptive_ode**, and the CLI **adaptive-solver**. This milestone builds distributions but does not claim a public PyPI publication.

## Known equations

```python
from adaptive_ode import BenchmarkConfig, benchmark

result = benchmark("mismatch", config=BenchmarkConfig(num_points=301))
print(result.selected_method)
print(result.ranking)
result.save("runs/my-benchmark")  # new directory required
```

Built-in keys: `lorenz`, `vanderpol`, `robertson`, `mismatch`. Pass a `DynamicalSystem` for a custom Python RHS. Candidate integrators use its `candidate_rhs`; references use its true `rhs`. The special built-in mismatch reference is analytical; other known-equation references are tight numerical approximations. User-supplied systems do not automatically get oscillator-specific calibration.

Default candidates are RK4, RK45, DOP853, BDF, Radau, LSODA; mismatch also has linear-oscillator parameter calibration. `include_ml=True` adds the trajectory residual NN, with two validation-tuned regularization settings. Install `[ml]` first. Physics-informed/PyTorch and the old Neural ODE modules are excluded from the supported API; their audit findings remain unresolved experimental code.

## Observations only

```python
from adaptive_ode import benchmark_data

result = benchmark_data(times, states)  # states: (n_times, n_states), or 1D
```

Times must be finite, unique and strictly increasing. All observations must be finite. Defaults are cubic spline extrapolation and a degree-two learned ODE; optional ML adds a direct time-to-state MLP. The polynomial model uses finite differences and ridge regression, not sparse SINDy. Test observations can be noisy. Cubic spline extrapolation can fail badly beyond training times, and remains a baseline rather than a recommended universal forecast model.

## Configuration and evaluation contract

`BenchmarkConfig` fields:

| Field | Default | Meaning |
|---|---:|---|
| train_fraction | .6 | Initial chronological training block. |
| validation_fraction | .2 | Next block for variant/method selection. |
| num_points | 301 | Generated points for built-in/custom equation benchmarks. |
| seed | 7 | Neural initialization seed. |
| include_ml | False | Add optional neural candidates to default sets. |
| nn_max_iter | 1200 | Maximum neural training iterations. |
| max_nfev | 50000 | Maximum RHS calls per prediction integration. |
| train_stride | 1 | Retain every nth point within the training block. |

At least two samples must remain in every block and in the retained training set. There is no refitting on validation data. Selection/ranking uses observed validation MSE only. Final-test errors, runtimes and descriptive residuals are reports, not selection criteria. Candidate predictions are checked over the full grid for finite complete output; future integration failure makes a candidate ineligible even without looking at future labels. This is a numerical robustness check, not test-error tuning.

MSE/RMSE/MAE weight state dimensions equally in original units. No cross-system aggregate score or automatic unit normalization is claimed. Prediction starts at the original initial state; there is no true-state reset at validation/test boundaries. A single time-series split is not an independent new-trajectory test.

## Custom candidates

```python
import numpy as np
from adaptive_ode import Candidate, FittedModel, evaluate

def fit_constant(t_train, y_train):
    mean = y_train.mean(axis=0)  # receives training data only
    return FittedModel(lambda t: np.tile(mean, (len(t), 1)))

candidate = Candidate("constant", "baseline", fit_constant)
result = evaluate(times, states, [candidate])
```

`Candidate(name, family, fit, group=None)` defines a trial. Names must be unique. Trials sharing a group are treated as variants of one method and selected on validation MSE. All attempts remain in `result.trials`. Fitting callbacks must not capture future labels through closures or external state; the framework cannot police arbitrary user code.

`FittedModel(predict, details={}, transfer=None)` supplies a times-only predictor, JSON-serializable metadata, and an optional `(times, new_initial_state)` callback. Models with a transfer callback expose `predict_from_initial`; trajectory-only models raise `NotImplementedError`. Transfer performance requires separate held-out trajectories; it is not implied by a low original-trajectory score.

Additional supported factories in `adaptive_ode.methods`: `classical_candidate`, `oscillator_calibration_candidate`, `polynomial_correction_candidate`, `spline_candidate`, `learned_polynomial_candidate`. Optional factories in `adaptive_ode.ml_methods`: `trajectory_residual_candidate`, `acceleration_residual_candidate`, `mlp_trajectory_candidate`. Acceleration models require two observed states (position, velocity), uniform training times, and enough points for the chosen smoothing window. They are oscillator-specific extensions, not arbitrary-system model discovery.

## Results and exports

- `selected_method`: group name chosen on validation, or None when all methods fail.
- `ranking`: copies of method rows ordered by validation MSE; failures last.
- `rows`, `trials`: metrics, timing, chosen variant, warnings and failure messages.
- `test_status`: records final-scoring success separately. Overflowing final metrics remain unavailable and never trigger selection of a different winner.
- `models`: retained fitted models, keyed by method group; no automatic model serialization.
- `predictions`, `target`, `observed`, `t`: numeric arrays.
- `split`: training/validation/test indices plus actual retained `fit` indices.
- `save(path)`: results.json, metrics.csv, predictions.npz. Refuses overwrites. Models are not pickled; custom model objects require refitting to recover them.
- `plot(state=0)`: returns a Matplotlib figure; install `[plots]`. Shows the validation boundary and final test region.

Fit time and full-grid prediction time are measured separately. `search_seconds` includes all variants' fit and prediction costs. These are single wall-clock observations, not performance confidence intervals. Use controlled repeated timing for speed claims.

Custom string equations (the `custom` extra) accept arithmetic and allowed one-argument functions, converted from a restricted AST to SymPy without evaluating Python source. Attribute access, indexing, comprehensions and arbitrary calls are rejected. Numeric exponents have magnitude <=16; expressions have at most 150 AST nodes; numeric constants have magnitude <=1e12. Variables/parameters must be unique, non-overlapping and cannot shadow `t` or function/constant names. For more general trusted equations, pass a Python RHS through `DynamicalSystem`.
