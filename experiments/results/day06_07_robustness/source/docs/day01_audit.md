# Day 1 audit: scientific evaluation and reproducibility

Baseline Git revision: `26a1502` (working tree initially clean). This is a local code audit, not a claim of published-library novelty or independent validation of all solver families.

## Findings

| Priority | Finding and evidence | Consequence | Days 1–5 disposition |
|---|---|---|---|
| High | `pipeline/run_pipeline.py` selects the minimum held-out MSE and reports that same value. Both known-equation and uploaded-data paths use only two splits. | Selection performance is not an untouched estimate of generalization. | A separate experiment path has train/validation/test blocks and freezes selection before final scoring. Legacy app behavior remains explicitly exploratory. |
| High | `_mask_train_test` randomly samples time points across a single known-equation trajectory. | This measures reconstruction between observed times, not future forecasting. | Research study uses chronological blocks; raw app results are not relabeled as forecasting evidence. |
| High | No parameter-calibration candidate exists in the app; its mismatch example only changes damping and stiffness. | Neural correction can appear necessary when ordinary parameter fitting would suffice. | Add bounded training-only parameter fitting and compare it fairly. |
| High | `run_pinn_surrogate` uses true oscillator coefficients in its mismatch RHS, while classical and hybrid baselines use wrong coefficients. | That mismatch tournament gives PINN privileged physics information. | Exclude PINN from the controlled study and record the limitation; repair before a later fair PINN comparison. |
| High | Legacy `solvers/neural_ode.py` discards supplied observations (`del y_noisy_context`) and regenerates synthetic clean training trajectories. | It is unsuitable as evidence for learning unknown dynamics from supplied observations. | Not called by the current app tournament or new study; exclude from validated-method claims pending redesign. |
| Medium | Classical predictions are compared against a tight-tolerance numerical solve. | The reference has numerical error and is not independent analytical truth. | Use analytical truth for the oscillator; check against matrix exponential, DOP853, and RK4 convergence. |
| Medium | `choose_best` uses MSE only; diagnostics are computed afterward, across the full trajectory. | Runtime/diagnostics do not drive selection, and diagnostics mix fitted and held-out points. | Research results separate validation selection, test-only descriptive residuals, and timings. |
| Medium | `_recommendation_reason` attributes improvements to physics penalties or systematic error without ablation. | A low score alone does not establish the explanation. | New report uses measured comparisons and qualified conclusions; app explanation is not treated as scientific evidence. |
| Medium | The app fixes the hybrid seed to 7; PINN has no explicit torch seed. | Stability across training initializations is unknown. | Repeat the controlled residual NN across five declared seeds. PINN reproducibility remains future work. |
| Medium | App runtime combines training, integration, prediction, and metric computation. Reference cost is outside each candidate's time. | These are not clean training/inference or complete workflow cost comparisons. | Record fit, prediction, tuning-search, and end-to-end study costs separately. |
| Medium | Existing requirements only have lower bounds; no saved machine-readable configs, versions, raw benchmark results, or test suite. | The PNGs alone do not establish reproducibility. | Create isolated research environment, exact lock, configs, source snapshots, raw results, and focused verification tests. |
| Medium | Diagnostic tests can suppress exceptions, pool state dimensions for some tests, and use a fixed state-dependence threshold. | A displayed "Pass" is not always evidence that a test was applicable and passed. | Do not rely on these flags for research conclusions; retain per-state descriptive residual measures. |
| Low | `save_plots=False` disables writing PNGs but still creates figures. | It is not a compute-saving headless mode. | New experiment runner creates plots only after numeric runs. |

## What the existing implementation does correctly

The hybrid and uploaded-data scalers/estimators fit only their designated training rows. Computing residual arrays before selecting training rows is not by itself label leakage: the app fits the hybrid on indexed training residuals. The main evaluation defect is reuse of the held-out set for selection and reporting. Failed/non-finite candidate predictions are already excluded from recommendations.

## Baseline capture

`python -m adaptive_ode.experiments.capture_baseline --output NEW_DIRECTORY` runs the original mismatch pipeline with 300 points, training fraction 0.7, seed 7, no noise and no PINN, without overwriting the app's tracked output figures. It saves the ranking, environment, source, reference trajectory, and figures. Its held-out score is an exploratory baseline, not the new experiment's final-test result.

See `experiments/results/day01_baseline/baseline.json` for the actual captured values. The analytical-reference discrepancy is measured there. PyTorch/torchdiffeq and the interactive Streamlit UI are not exercised in the Days 1–5 environment.

## Scope boundary

The new research workflow addresses the evaluation defects for the controlled oscillator experiment; it does not silently certify or replace the original app, parser, PINN, or legacy Neural ODE implementation. Broader system testing, noise/missing-physics studies, public API, packaging, and CI remain later milestones.
