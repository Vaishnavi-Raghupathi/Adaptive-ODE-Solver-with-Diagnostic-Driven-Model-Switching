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

Times must be finite, unique and strictly increasing. All observations must be finite. Defaults are persistence, a linear time trend, cubic spline extrapolation and degree-one/two learned ODEs; optional ML adds a direct time-to-state MLP. The ODE models use finite differences and ridge regression, not sparse SINDy. Test observations can be noisy. Cubic spline extrapolation can fail badly beyond training times, and remains a baseline rather than a recommended universal forecast model.

Pass `rhs=callable` to additionally compare RK4, RK45, DOP853, BDF, Radau and LSODA against the same observations. State order and units must match. The initial observation initializes integration; supplied parameters are not fitted automatically. Do not combine `rhs` with explicit `candidates`. See the [dataset workflow](dataset_workflow.md) for equation files and comparison limitations.

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
| selection_metric | "mse" | "mse" or "normalized_rmse"; normalization uses retained training states only. |
| selection_policy | "min_error" | Lowest error, or "fastest_within_target". |
| accuracy_target | None | Required nonnegative maximum validation score for fastest_within_target. |
| selection_cost | "prediction_seconds" | Median prediction time, or "fit_plus_prediction_seconds". |
| timing_repeats | 1 | Repeat predictions 1-20 times using one fitted model; summarize by median. |
| solver_sweep | False | Preset adaptive-solver tolerance and RK4 step-size variants. |

At least two samples must remain in every chronological block and in the retained training set. There is no refitting on validation data. Variant and method selection use the declared policy: either minimum validation error or minimum measured cost among candidates meeting the validation accuracy requirement. No qualifying method produces no recommendation. Ranking rows remain ordered by validation score, so the chosen method need not be first when cost determines the choice. Test errors and descriptive test residuals never select a method. In the chronological workflow candidates are checked over the full grid for finite complete output; future integration failure makes a candidate ineligible even without looking at future labels. The independent-experiment workflow freezes selection before predicting test records.

MSE/RMSE/MAE weight state dimensions equally in original units. Optional normalized RMSE divides state errors by retained training SD before taking pooled RMSE; constant states use their maximum absolute training magnitude, or one at zero. The default API metric remains MSE. No cross-system aggregate score is claimed. Prediction starts at the original initial state; there is no true-state reset at validation/test boundaries. A single time-series split is not an independent new-trajectory test.

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

`Candidate(name, family, fit, group=None, fit_trajectories=None)` defines a trial. Names must be unique. Trials sharing a group are treated as variants of one method and selected using the configured policy. All attempts remain in `result.trials`. The optional `fit_trajectories(times_list, states_list)` callback receives full training experiments for independent evaluation; its fitted model requires a transfer callback. Fitting callbacks must not capture future labels through closures or external state; the framework cannot police arbitrary user code.

`FittedModel(predict, details={}, transfer=None)` supplies a times-only predictor, JSON-serializable metadata, and an optional `(times, new_initial_state)` callback. Models with a transfer callback expose `predict_from_initial`; trajectory-only models raise `NotImplementedError`. Transfer performance requires separate held-out trajectories; it is not implied by a low original-trajectory score.

Additional supported factories in `adaptive_ode.methods`: `classical_candidate`, `oscillator_calibration_candidate`, `polynomial_correction_candidate`, `spline_candidate`, `learned_polynomial_candidate`. Optional factories in `adaptive_ode.ml_methods`: `trajectory_residual_candidate`, `acceleration_residual_candidate`, `mlp_trajectory_candidate`. Acceleration models require two observed states (position, velocity), uniform training times, and enough points for the chosen smoothing window. They are oscillator-specific extensions, not arbitrary-system model discovery.

## Results and exports

- `selected_method`: group name chosen on validation, or None when all methods fail.
- `ranking`: copies of method rows ordered by configured validation score; failures last.
- `assessment`: training-data profile, baseline comparison, empirical accuracy/runtime Pareto set and limitations. No future state values inform this recommendation.
- `report()`: portable Markdown recommendation, metrics, warnings and scope.
- `rows`, `trials`: metrics, timing, chosen variant, warnings and failure messages.
- `test_status`: records final-scoring success separately. Overflowing final metrics remain unavailable and never trigger selection of a different winner.
- `models`: retained fitted models, keyed by method group; no automatic model serialization.
- `predictions`, `target`, `observed`, `t`: numeric arrays.
- `split`: training/validation/test indices plus actual retained `fit` indices.
- `save(path)`: results.json (schema 2), metrics.csv, predictions.npz and report.md. Refuses overwrites. Models are not pickled; custom model objects require refitting to recover them.
- `plot(state=0)`: returns a Matplotlib figure; install `[plots]`. Shows the validation boundary and final test region.

Fit time and full-grid prediction time are measured separately. Prediction time is a median over configured repeats; `search_seconds` includes all variants and repeated prediction work. Fit-plus-prediction selection adds one fit, whose uncertainty is not repeatedly estimated. Predictors must return consistent outputs across repeats. Use controlled repeated benchmarks for broad speed claims.

## Independent trajectories

`Trajectory(t, states, name)` holds an experiment. `benchmark_trajectories(train=[...], validation=[...], test=[...], config=..., rhs=None, candidates=None)` fits full training records, selects on separate validation records, and only then predicts separate test records. Exact duplicate content and reused names are rejected. `rhs` and explicit candidates are mutually exclusive. Different records can have different lengths, time grids and initial states. Learned ODE derivatives are formed separately within each training record.

`TrajectoryBenchmarkResult` exposes `ranking`, `assessment`, `report()`, `save()`, `models`, per-trial/per-episode diagnostics, original datasets and prediction arrays. Schema-3 exports include all records, role membership, hashes, normalization, metrics, environment information and whole-test-trajectory bootstrap intervals conditional on fixed models. The first observed state initializes each held-out rollout and is excluded from scores. See the [full protocol](independent_experiments.md), including limitations and a runnable manifest example.

Custom string equations (the `custom` extra) accept arithmetic and allowed one-argument functions, converted from a restricted AST to SymPy without evaluating Python source. Attribute access, indexing, comprehensions and arbitrary calls are rejected. Numeric exponents have magnitude <=16; expressions have at most 150 AST nodes; numeric constants have magnitude <=1e12. Variables/parameters must be unique, non-overlapping and cannot shadow `t` or function/constant names. For more general trusted equations, pass a Python RHS through `DynamicalSystem`.
