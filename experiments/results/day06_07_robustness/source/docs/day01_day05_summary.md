# Days 1–5: completion and initial findings

## Main finding

For this noise-free oscillator with incorrect damping/stiffness, fitting those two physical parameters is sufficient. The current trajectory residual NN does not improve on that baseline when evaluated on future times. This is evidence for calibrating a structurally correct model before adding a neural correction, within this experiment's narrow assumptions.

At full mismatch, future test MSE (mean over five NN initialization seeds) is:

| Approach | Test MSE |
|---|---:|
| Assumed RK4 | 0.8754 |
| Assumed DOP853, standard or tight tolerance | approximately 0.8754 |
| Training-only parameter fitting + DOP853 | 4.691 × 10^-19 |
| RK4 + trajectory residual NN | 10.18 ± 2.05 |

The very small calibrated error reflects an exactly specified, fully observed, noise-free model with known initial state. It is not an expected real-world accuracy. The oracle and calibrated predictions numerically match the analytical reference at the plotted scale.

## Work completed

| Day | Delivered | Evidence |
|---|---|---|
| 1 | Implementation audit, isolated Python 3.13 environment, dependency lock, original pipeline capture. | [Audit](day01_audit.md), [baseline record](../experiments/results/day01_baseline/baseline.json), [lock](../requirements-research.lock). |
| 2 | Written protocol and fixed production configuration before running the study. | [Protocol](experiment_protocol.md), [configuration](../experiments/configs/day05_mismatch.json), copies/hashes in the run's source snapshot. |
| 3 | Analytical reference, checked RK4/DOP853 baselines, bounded training-only parameter calibration. | [Implementation](../adaptive_ode/experiments/oscillator.py), [verification tests](../tests/test_research_baselines.py). |
| 4 | One-command runner with validation-only selection, separate timings, preserved failures/warnings, exact source snapshots and raw predictions. | [Runner](../adaptive_ode/experiments/mismatch.py), [reproduction guide](../experiments/README.md). |
| 5 | Completed five mismatch levels × five seeds: 25 cases and 50 NN fits, without changing settings after final test scores. | [Generated report](../experiments/results/day05_mismatch/report.md), [CSV](../experiments/results/day05_mismatch/metrics.csv), [provenance](../experiments/results/day05_mismatch/provenance.json). |

## Evaluation and verification

- Chronological split: 240 training samples, 80 validation samples, 81 final test samples. Validation chooses NN alpha and recommended method; final test data do not change either.
- Correct-model control: tight DOP853 selected in all five runs.
- Nonzero parameter mismatch: fitted DOP853 selected in all 20 runs.
- The hybrid meets the predeclared practical-improvement criterion in zero seeds at every level. It modestly lowers raw future error versus assumed RK4 at the smallest nonzero mismatch on average, but it does not beat calibration.
- No failed attempts or NN convergence warnings in the production run. A lack of convergence warnings is not a guarantee of a good fit or extrapolation.
- 28 focused tests passed, including independent matrix-exponential agreement, fourth-order RK4 convergence, recovery of known parameters, input validation, future-label poisoning, validation/training separation, failure retention, and artifact creation.
- The first verification pass exposed an overly strict 2e-14 matrix-exponential comparison tolerance (differences up to 3e-13). A 1e-12 roundoff tolerance resolved it before the production run; the reference formula was unchanged. This tolerance is far below the numerical integration tolerances.
- The original pipeline capture completed. Its tight numerical reference differs from the analytical solution by at most 5.34e-10 in absolute state value.
- Reproduction from the saved source in an isolated temporary directory is recorded in [day05_reproduction_check.json](../experiments/results/day05_reproduction_check.json). It compares selections, validation/test scores, fitted details, NN loss curves, and all saved trajectory arrays; timings are intentionally excluded.
- Python compilation, dependency consistency, and whitespace checks passed. The Streamlit UI and PyTorch candidates were not exercised in this research environment; application source and existing plots were not changed.

## Why the original demo result differs

The captured app tournament recommends the hybrid with held-out MSE 0.1869 versus approximately 1.7309 for classical methods. That experiment has randomly held-out points across the same trajectory and no parameter-fitting candidate. The research study uses future segments, a different sample count, and a stronger baseline. These scores are not a controlled before/after comparison; they evaluate different tasks.

The new result does not invalidate the app's measured reconstruction scores. It limits the claim: the existing time-conditioned trajectory correction should not be advertised as a generally better dynamics model or a demonstrated future forecasting method.

## Next scientific work

Day 6 should add a model with a missing physical term, plus noisy and reduced-data settings, under a new written protocol. Parameter fitting must remain in the comparisons. Day 7 should examine generalization beyond the training trajectory where supported. Preserve the Day 5 negative result as a control rather than tuning on its final test set until the NN wins.

Packaging, a stable public API, CI/CD, and public distribution have not started. The first five days deliver a reproducible evaluation foundation and evidence that will guide that scope.
