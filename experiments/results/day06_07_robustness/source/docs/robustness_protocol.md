# Days 6–7 protocol v2: noise, missing physics, and new initial states

Written before the full v2 experiment. Day 5 data/results remain unchanged. A new physical problem is included because parameter mismatch alone is resolvable by calibration; this is not tuning against the old final test scores.

## Fixed design

Use `experiments/configs/day06_07_robustness.json`. Two systems have x'=v and acceleration `-0.1v - 4x - beta*x^3`, with beta=0 (parameter-only control) or beta=0.5 (missing nonlinear restoring force). All candidates start with assumed linear acceleration `-0.5v - 3x`; candidates receive neither beta nor clean derivatives. The true nonlinear RHS is used only to generate evaluation targets. The polynomial library intentionally includes the true cubic term; report that advantage explicitly.

There are 2 systems × 3 Gaussian observation-noise standard deviations (0, .02, .1 in raw state units) × 2 retained-training strides (1,4) × 5 seeds (101,202,303,404,505) = 60 cases. Seeds control both observation noise and neural initialization; trajectories without noise remain deterministic. Both position and velocity are observed. Initial state `[2,0]` is known exactly and not perturbed. Noise is added once to observations for training and validation; test error is reported against clean truth. Use the same noisy observations for all candidates in a case, and record error against noisy test observations separately if needed.

Generate 401 uniform times in [0,20]. Training block is t=0–11.95, validation t=12–15.95, final future test t=16–20. Retain every fourth point of the training block in the reduced-data condition (60 versus 240 fitting points). Do not subsample validation or test. Record exact fitting indices. No retraining on validation data. Use only noisy validation MSE to choose variants and the recommended method. Clean test targets and held-out initial states cannot affect selection.

## Prespecified methods

1. Assumed RK4 and assumed DOP853: no parameter learning.
2. Calibrated linear oscillator: bounded least squares on the retained noisy training trajectory, same bounds/budget as Day 5. It cannot introduce a missing cubic force.
3. Existing trajectory residual NN: time + RK4-state features; two 64-unit tanh layers, Adam, 1,200 iterations maximum, alpha in [1e-4,1e-2]. Validation selects alpha. The RK4 baseline uses substeps no larger than .05 at both fitting and prediction, avoiding a change of numerical baseline when observations are thinned. It is not eligible for new-initial-state transfer.
4. Polynomial acceleration correction: regress training acceleration-minus-assumed-physics on `[x,v,x^3,x^2v,xv^2,v^3]`, ridge alpha=1e-6, training-only feature scales. Estimate acceleration from observed velocity using a third-order Savitzky–Golay filter. Validation selects window 9 or 21. Exclude the half-window boundaries before regression. This is an interpretable fixed-library baseline, not sparse structure discovery.
5. State-based neural acceleration correction: same NN architecture/alpha choices, training-only state scaling and residual-target scaling. Estimate training states/acceleration with the same filter at fixed window 9; exclude boundaries. Integrate the resulting corrected differential equation rather than correcting a precomputed trajectory.

Derivative estimation is a weak point under noise and sparse sampling. This study records that sensitivity rather than treating supplied clean derivatives as training data. Fixed-library polynomials may outperform neural networks; retain that result. NN convergence warnings, variants and failures remain visible. All integrations have an RHS-evaluation budget of 50,000; exhaustion is a failed prediction, not a discarded case.

## Transfer evaluation

After variants and the recommended method are frozen on the original trajectory, use the same fitted models on three unseen initial states `[1,0]`, `[3,0]`, `[2,1]`, generating new clean targets. Report full-horizon [0,20] MSE for each initial state and method, separately from original-trajectory future MSE. No re-fitting, boundary resets, validation on the new trajectories, or per-initial-state variant selection is permitted. The time-conditioned trajectory NN is marked `not_supported`; do not pretend it learned a transferable vector field.

## Reference, interpretation and artifacts

Linear truth uses the analytical solution. Nonlinear truth uses DOP853 at rtol=1e-11/atol=1e-13 and is cross-checked against Radau at the same tolerance for every initial state. Save discrepancies. Test both reference construction and model code with small software fixtures before production; do not use final-test performance to change the frozen configuration.

Report raw MSE/RMSE/MAE, per-state residual summaries, validation selection counts, all fitting/prediction/search costs, warnings, and failures. Aggregate by system/noise/stride/method using mean and sample SD over seeds; do not pool different systems or claim statistical significance from five seeds. Count a candidate as practically better than calibration only if it lowers future test MSE by >=10% and >=1e-6 in >=4/5 seeds of a condition, including failures as non-wins. For transfer, report the same criterion separately per held-out initial state. This practical threshold is not a statistical test. Never rank or select a method using final-test or transfer error.

Use single-thread BLAS. Save exact source/config/environment/provenance, NPZ arrays and all trial JSONs. The representative plot is the first seed, missing-cubic system, noise=.02, stride=1; this choice is fixed before results. Keep failures and negative findings. If a condition is unsuitable, explain it without retroactively changing budgets.
