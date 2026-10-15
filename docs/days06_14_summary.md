# Days 6–14: library release candidate and scientific assessment

## Decision

Yes: this project is now a pip-installable Python library, **adaptive-solver-bench 0.1.0**, with import package **adaptive_ode**. Its defensible purpose is reproducible comparison of numerical integration, calibrated physics, and learned corrections. The experiments do not support positioning it as a universally better neural solver.

The local deliverable includes a wheel and source distribution in dist/, a public API and CLI, three tutorials, a Streamlit client, tests, and CI/manual release workflow files. Public PyPI publication and successful remote CI execution remain external release steps, not claimed completed here.

## Milestones delivered

| Day | Delivered |
|---|---|
| 6 | Written v2 protocol; 60 cases covering parameter mismatch versus missing cubic force, three noise levels and two training densities. |
| 7 | Three predetermined unseen initial states per fitted model; no refitting or new-initial-state tuning; reference cross-checks and archived-source reproduction. |
| 8 | Lorenz, Van der Pol, Robertson and mismatch controls, three deterministic timing repeats each, reference disagreement and failed predictions recorded. |
| 9 | Public benchmark/benchmark_data/evaluate API, explicit candidate/model protocol, chronological splits and structured results. |
| 10 | Standard package metadata, optional dependency extras, installed CLI, wheel and source builds. Core dependencies are NumPy and SciPy only. |
| 11 | 71 passing tests covering numerical behavior, label separation, selection, failures, exports, optional methods, CLI, custom expressions and demo execution. |
| 12 | GitHub CI matrix and manual artifact/TestPyPI/PyPI release workflows configured and YAML parsed locally. Remote jobs remain pending. |
| 13 | Three runnable tutorials and synthetic sample CSV, API/release documentation, and Streamlit interface using the public library. Original app preserved in legacy_app.py. |
| 14 | Local metadata verification and clean-environment installed-wheel checks; alpha release prepared, not uploaded. |

## Additional scientific findings

The [robustness report](../experiments/results/day06_07_robustness/report.md) contains every condition and its failures. Across 60 cases, validation selected calibration in 34 and polynomial correction in 26. Neither neural method was selected. Calibration won all 30 parameter-only cases, including noisy/reduced-data settings.

For the missing-cubic system with dense training data:

| Method | Clean future test MSE | Noise std .02: mean future test MSE |
|---|---:|---:|
| Assumed RK4 | 1.044 | 1.044 |
| Calibrated linear oscillator | 1.407 | 1.407 |
| Polynomial acceleration correction | 9.172e-6 | 3.228e-3 |
| Time-conditioned trajectory residual NN | 3.777 | 3.806 |
| State-based neural acceleration correction | 1.171 | 1.184 |

Values average five seeds. Clean deterministic baselines do not gain independent physical observations from repeated seeds. Both states and the initial condition are known; see the [frozen protocol](robustness_protocol.md).

The polynomial correction met the predeclared practical criterion against calibration in five of six missing-physics noise/density conditions. In the noisiest, sparsest condition it qualified in only 3/5 seeds, below the required 4/5. The state-based NN qualified in one condition (clean, dense), but still lost validation selection to the polynomial method. The original time-conditioned NN qualified in no condition.

The polynomial library explicitly includes the true missing cubic term; this is a favorable structured baseline, not arbitrary physics discovery. It suffered 21 unsuccessful variants, leaving two case-level fits without any usable polynomial variant. Transfer results include 890 successful predictions, 180 not_supported entries for the time-conditioned NN, six fit_failed entries, and four failed transfer predictions. Failures are retained rather than removed.

On clean dense missing-physics data, polynomial full-horizon MSE at unseen initial states was approximately 1.80e-7 for [1,0], 2.63e-4 for [3,0], and 1.09e-5 for [2,1], without refitting. That is limited evidence for this model/system, not unrestricted generalization.

Reference cross-check differences were at most approximately 1.14e-10 in absolute state value for the robustness suite. Replaying the archived source on the same machine reproduced all 60 cases, selections, scores, fit details, transfer metrics and 2,009 numeric arrays bit for bit; timings were excluded. See [reproduction evidence](../experiments/results/day06_07_reproduction_check.json).

The [classical controls](../experiments/results/day08_classical/report.md) selected Radau for Lorenz, Van der Pol and Robertson, and calibration for mismatch. Under the stated grid, derivative guards and 50,000-RHS-call budget, Robertson's RK4, RK45 and DOP853 attempts failed; this does not establish that those algorithms cannot solve Robertson with other settings. Repeats measure timing only. Alternate-reference disagreement was below 9e-10. No pooled cross-system accuracy or universal ranking is claimed.

## Library scope

- Known-equation and observations-only workflows, plus a custom candidate interface.
- Training-only fits, validation-only variant/method selection, and separate future test reports.
- Full-grid finite prediction checks establish eligibility. Final-score overflow is recorded separately and never causes reselection.
- Numeric NPZ/JSON/CSV exports with splits, configurations, errors, costs, warnings and failures. Fitted models are retained in memory, not serialized with pickle.
- State-based oscillator corrections can predict from new initial states. Trajectory-only models explicitly cannot.
- The new demo calls the public API; the original app and requirements are preserved for reference.
- Custom strings are converted from a restricted arithmetic AST instead of evaluating Python expressions; general trusted equations can use Python RHS callables.
- Legacy PINN/Neural ODE modules remain outside the validated API. Packaging does not erase their audit findings.

## Verification and release boundary

Evidence: [test XML](../experiments/results/release_verification/pytest.xml), [core wheel checks](../experiments/results/release_verification/wheel_core.json), [optional-extra checks](../experiments/results/release_verification/wheel_extras.json), and [final artifact hashes/checks](../experiments/results/release_verification/artifacts.json).

Verification uses Python 3.13 on macOS ARM64. The base wheel was installed in a fresh environment without scikit-learn, Streamlit or PyTorch and exercised outside the checkout. The installed CLI and all three tutorials passed. Optional-extra checks exercised neural fitting, plotting, custom equations and the bundled Streamlit demo. Tests do not certify every branch of retained legacy code; coverage is not 100%.

The configured CI matrix targets additional Python/Linux combinations but has not run remotely here. No source push, remote tag, GitHub release or PyPI upload was performed. A PyPI lookup returned 404 for the proposed name; it is not reserved or guaranteed. See the [release guide](releasing.md) and [official publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

## Most useful next work

The state-based correction is already a meaningful extension. Next prioritize multi-trajectory fitting with trajectory-level testing, stability-constrained/integral-form corrections, and cost-aware validation recommendations. Those address measured weaknesses; adding architectures or inflating method counts provides less evidence. See [extensions.md](extensions.md) for acceptance criteria.
