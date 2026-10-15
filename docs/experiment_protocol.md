# Days 2–5: frozen parameter-mismatch protocol (v1)

This protocol and `experiments/configs/day05_mismatch.json` are written before the full Day 5 run. Changes motivated by final test scores require a new protocol and fresh final evaluation; do not overwrite the saved run. Small test fixtures verify software behavior and are not used to tune the production study.

## Question and scope

Does the existing RK4 + trajectory residual neural network add value beyond numerical integration and parameter fitting when an oscillator has incorrect damping and stiffness? This first experiment is intentionally noise-free. Noise, missing physical terms, other systems, and new initial conditions belong to subsequent work.

## System and observations

The true dynamics are `x' = v`, `v' = -0.1 v - 4 x`, with initial state `[2, 0]`, t in `[0, 20]`, and 401 equally spaced samples. Both state dimensions are observed in original units; the initial condition is known exactly. A closed-form underdamped solution generates truth, independently checked against a matrix exponential and numerical integration.

At mismatch fraction λ in `[0, .25, .5, .75, 1]`, the assumed model has damping `0.1 + 0.4 λ` and stiffness `4 - λ`. Thus λ=0 is a correct-model control and λ=1 reproduces the repository's original wrong oscillator. The two parameter errors change together; this is one mismatch direction, not an exhaustive sensitivity analysis.

## Data separation

- Training: indices 0–239, t=0–11.95 (60%).
- Validation: indices 240–319, t=12–15.95 (20%).
- Final test: indices 320–400, t=16–20 (remaining 81 points).

Models, scalers, and parameters are fit on training observations only. Validation chooses NN regularization and the recommended method using observed-state MSE. No retraining on validation data occurs. All predictions continue from the original known initial state; no true future states or boundary resets initialize validation/test predictions. Full prediction times are allowed as inputs. Final test targets are scored only after selection is frozen. Tests poison final-test labels to verify they cannot affect fitting, predictions, or selection, and alter validation labels to verify training is unaffected.

## Methods and budgets

| Method | Definition |
|---|---|
| Assumed RK4 | Existing fixed-step RK4, step 0.05, on the assumed equations. |
| Assumed DOP853 | Assumed equations, rtol=1e-6, atol=1e-8. |
| Assumed DOP853 tight | Same equations, rtol=1e-10, atol=1e-12; tests whether tighter integration fixes model error. |
| Fitted DOP853 | Fit damping and stiffness to the training trajectory with bounded `least_squares`, then integrate at rtol=1e-9, atol=1e-11. |
| RK4 + residual NN | Train an MLP on `[t, assumed RK4 states]` to predict observed-minus-RK4 trajectory residuals. |
| True-equation DOP853 oracle | rtol=1e-10, atol=1e-12; independent numerical check only, excluded from recommendations. |

Parameter fitting uses equal raw-state residuals; bounds `[0, 2]` for damping and `[0.5, 8]` for stiffness; one start at the assumed parameters; maximum 200 optimizer function evaluations; ftol/xtol/gtol=1e-10. Objective integrations use rtol=1e-9, atol=1e-11. An unsuccessful optimizer is recorded as a failure. Starting at the correct parameters in λ=0 is explicitly a favorable control, not a parameter-identification challenge.

The NN preserves the app's architecture: two 64-unit tanh layers, Adam, learning rate 0.001, maximum 1,200 iterations, training-only StandardScaler. Candidate alphas are fixed at `[1e-4, 1e-2]`. Early stopping with an internal holdout is disabled; training-loss tolerance is 1e-4 with 10 no-improvement iterations. Other explicit optimizer settings are in the runner and exact versions in the lock. Validation chooses the lowest-MSE alpha. There is no tuning against final test error. Convergence warnings, loss curves, and failed attempts are saved.

## Repetition and compute budget

Run 5 mismatch levels × 5 seeds `[7, 17, 27, 37, 47]` = 25 cases, with 50 NN fits maximum. Seeds vary neural initialization and minibatch order; all trajectories are deterministic and identical across seeds. Set BLAS to one thread. Run sequentially on the local CPU, with a fixed iteration/evaluation budget rather than an accuracy-dependent search. Do not interpret these seeds as independent physical datasets.

## Measures, costs, and interpretation

- Report validation MSE and final-test MSE/RMSE/MAE, averaged equally over both state dimensions in raw units.
- Report test residual mean, standard deviation, maximum magnitude, and lag-one correlation separately per state. These are descriptive summaries, not hypothesis tests or causal explanations.
- Record fit and full-grid prediction times separately, including RK4 baseline generation in NN cost. The NN family search cost includes both alpha fits and both trial predictions. Save total end-to-end study time including reference generation, artifact writing, and report creation, but excluding environment installation/module imports.
- Aggregate test MSE as mean and sample SD across seeds; retain all raw rows, convergence warnings, failures, and per-level validation selections. Timings are descriptive single-machine measurements and may include first-use effects, not statistically established speed rankings.
- Call the hybrid **practically helpful** at a level only if its test MSE is at least 10% below both assumed RK4 and fitted DOP853, with an absolute improvement at least 1e-8, in at least 80% (4/5) of seeds. This is a predeclared practical rule, not a significance test.
- The representative trajectory plot always uses the first configured seed at maximum mismatch; do not choose a visually favorable seed after inspecting results.

Any negative result is retained. The experiment cannot establish general superiority of neural corrections, solve unknown-equation selection without targets, or justify claims about noisy observations, missing physics, new initial conditions, or other system families.

## Artifacts and reproduction

Each run has a unique directory and refuses overwrites. Preserve config, environment, source snapshot and SHA-256 manifest, Git base revision and dirty status, platform/thread information, per-case JSON/NPZ predictions, tuning trials, CSVs, failures, figures, and a generated report. Numerical reproducibility uses the same source and environment; timing equality across machines is not expected.

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.lock
python -m pytest tests -q
python -m adaptive_ode.experiments.capture_baseline --output experiments/results/day01_reproduction
python -m adaptive_ode.experiments.mismatch --config experiments/configs/day05_mismatch.json --output experiments/results/day05_reproduction
```

The captured environment is Python 3.13 on macOS ARM64; other platforms are not yet verified. Days 1–5 do not require PyTorch or Streamlit. The web app still uses its original requirements and exploratory evaluation path.
