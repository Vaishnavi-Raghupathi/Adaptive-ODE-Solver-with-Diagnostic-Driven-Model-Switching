# Independent experiments and accuracy requirements

This workflow asks which tested configuration meets an accuracy requirement on new experiments. It fits on complete training experiments, selects using separate validation experiments, freezes its selection, and only then predicts final test experiments. The original chronological single-trajectory workflow remains available.

## Run a complete example

```bash
python examples/04_independent_trajectories.py --output runs/independent-example
```

The example creates nine synthetic noisy oscillator experiments with different initial states, three per role. It compares six integrators at preset tolerances/steps, a learned linear ODE, a learned quadratic ODE and an initial-state-hold baseline. It exports reports, raw arrays, the example CSVs and a reusable manifest. This is a software demonstration, not independent real-data validation.

Repeat the exported dataset through the CLI:

```bash
adaptive-solver trajectories runs/independent-example/manifest.json \
  --policy fastest_within_target --accuracy-target 0.05 \
  --metric normalized_rmse --solver-sweep --timing-repeats 3 \
  --output runs/independent-replay
```

Timing-based selections can vary between runs or hardware. Deterministic numerical predictions and scores do not imply deterministic timing ranks. The Streamlit interface exposes the same workflow under **Independent trajectories** with an included example or separate CSV upload lists for each role.

## Python API

```python
from adaptive_ode import BenchmarkConfig, Trajectory, benchmark_trajectories

train = [Trajectory(t1, states1, "experiment-1"),
         Trajectory(t2, states2, "experiment-2")]
validation = [Trajectory(t3, states3, "experiment-3")]
test = [Trajectory(t4, states4, "experiment-4")]

config = BenchmarkConfig(
    selection_metric="normalized_rmse",
    selection_policy="fastest_within_target",
    accuracy_target=0.05,
    selection_cost="prediction_seconds",
    timing_repeats=3,
    solver_sweep=True,
)
result = benchmark_trajectories(train=train, validation=validation, test=test,
                                config=config, rhs=my_rhs)  # omit rhs if unknown
print(result.selected_method)
print(result.report())
result.save("runs/my-experiments")
```

Roles are explicit. `train_fraction` and `validation_fraction` do not apply. `train_stride` thins each training record separately. Every record must have increasing finite times, finite states, the same state dimension and meaning, and a unique name. Exact duplicate contents are rejected. This check cannot establish physical independence or detect all overlapping/reprocessed windows; construct splits at the level of independently acquired experiments, machines or sessions as appropriate.

The first observed state initializes each validation/test rollout. All subsequent values are evaluation targets. The first sample is excluded from metrics. No parameter updates or state resets occur within that rollout. Time grids may have different lengths and start times; numerical/learned RHS functions must handle their actual time convention. Explicit time dependence is meaningful only when time origins are defined consistently.

## Scores and selection

Each experiment contributes equally, irrespective of sample count. For MSE, average the per-experiment MSEs. For normalized RMSE, calculate per-experiment normalized MSE, average it across experiments, then take the square root. State scales are computed once using concatenated retained training observations. Longer training records therefore affect scale estimation more; the learned ODE regression itself assigns equal total weight to each training record. Derivatives are calculated within each training record, never across concatenation boundaries.

`min_error` selects the lowest validation score. `fastest_within_target` selects the lowest measured cost among configurations whose validation score is at most `accuracy_target`. The target is in the units of the configured score; it is an application requirement, not a learned noise estimate or future guarantee. No qualifying method produces **no recommendation**, rather than silently returning an inaccurate fallback. An accuracy target requires the corresponding policy.

Variants sharing a method group are selected by the same rule before choosing a method. If no variant qualifies, the lowest-error representative is displayed, flagged as not meeting the target. Group and global selection use no test targets. Test failure never triggers another choice, and methods with any failed test experiment receive no aggregate test score. Diagnostics retain individual successes and failures.

`prediction_seconds` is the median of repeated predictions of the full validation batch, using a single fitted model. `fit_plus_prediction_seconds` adds one measured fit. The number of repeats is configurable from 1 to 20. Every repeat must return the same prediction within tight numerical tolerances; callbacks must not mutate fitted state. Fitting time is not repeatedly estimated, so fit-plus-prediction cost has additional uncertainty. `search_seconds` includes all variants and repeated prediction work. Predicting final test experiments and calculating bootstrap intervals are not included in selection cost. Timing repeats are not statistical replications of the underlying experiment.

Preset solver sweeps use rtol 1e-3, 1e-5 and 1e-7 with atol=rtol*1e-2 for adaptive integrators. RK4 uses maximum internal steps based on the largest median retained training timestep, divided by 1, 2 and 4; it also subdivides at requested observation times. These are limited preset sweeps, not exhaustive optimization, and equal solver tolerances do not imply equal global errors. The interface displays empirical validation error versus cost for all successful variants.

Observation errors combine model mismatch, initial-state uncertainty, measurement noise and integration error. Comparing known-equation integrators with learned dynamics assesses candidate modeling pipelines; it does not isolate the numerical superiority of one algorithm. A pure integrator study needs appropriate trusted references and a work-precision experiment.

## Uncertainty and limitations

With at least two successful test trajectories, the report provides a percentile bootstrap interval from resampling whole test trajectories, keeping the fitted model and chosen configuration fixed. This describes variation across the supplied experiments under an independence assumption, not uncertainty from training, selection, calibration or future distribution shift. A few trajectories or identical observed scores can produce misleadingly narrow or degenerate intervals. A single test experiment receives no bootstrap interval.

This release does not yet implement rolling-origin evaluation, external-input learning, partial-state inference, trained neural ODE adapters or physically constrained hybrid learning on arbitrary systems. Unsupported time-to-state methods do not get silently adapted for initial-state transfer. Default `include_ml=True` is rejected for this protocol; provide an explicitly transferable custom model.

## Additional methods

```python
from adaptive_ode import Candidate, FittedModel

def fit_experiments(training_times, training_states):
    # Both arguments are lists of training arrays; no held-out observations.
    dynamics = fit_my_dynamics(training_times, training_states)
    def transfer(times, initial_state):
        return integrate_my_dynamics(dynamics, times, initial_state)
    return FittedModel(
        predict=lambda times: transfer(times, training_states[0][0]),
        transfer=transfer,
        details={"training_trajectories": len(training_times)},
    )

candidate = Candidate("my-method", "data-driven dynamics",
                      fit=lambda t, y: fit_experiments([t], [y]),
                      fit_trajectories=fit_experiments)
```

Arbitrary callbacks must not capture validation/test labels or other forbidden information from their environment. The library supplies permitted inputs but cannot sandbox trusted Python callbacks.

## CSV manifests and exports

```json
{
  "train": ["train-1.csv", "train-2.csv"],
  "validation": ["validation-1.csv"],
  "test": ["test-1.csv", "test-2.csv"],
  "time_column": "time",
  "state_columns": ["position", "velocity"],
  "equations": "equations.json"
}
```

Paths are relative to the manifest; omit `equations` when unknown. An entry may be an object with `path` and `name` to distinguish files sharing a basename. Select the same state order for all files. The CLI requires at least 12 rows per CSV. Equation files use the restricted JSON expression format documented in the dataset workflow.

`save()` refuses existing paths and exports schema-3 `results.json`, `metrics.csv`, `report.md` and safe numeric `trajectories.npz`. These include original records of different lengths, prediction arrays, split membership, content hashes, training scales, all validation trials, test episode diagnostics, bootstrap intervals, configuration and environment metadata. CLI metadata also records the manifest, source file hashes and JSON equation specification. Python callbacks/fitted models are not serialized; retain their source.
