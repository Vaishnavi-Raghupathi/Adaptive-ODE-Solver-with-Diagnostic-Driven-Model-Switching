# Changelog

## Unreleased - dataset comparison platform

- Independent-trajectory API and UI with explicit experiment roles, duplicate-content rejection and initial-state transfer.
- Accuracy-constrained variant/method selection with explicit no-recommendation outcomes.
- Repeated prediction timings, separate fit/prediction objectives and preset integrator tolerance/step sweeps.
- Equal experiment weighting, frozen selection before test prediction, and conditional whole-test-trajectory bootstrap intervals.
- Schema-3 exports, manifest CLI, a fourth runnable tutorial, extension documentation and contributor guidelines.

- Dataset-first Streamlit workflow with an example and optional equations for uploaded observations.
- Six numerical integrators compared against measured targets when an RHS is available.
- Persistence, time-trend and learned linear ODE controls alongside existing data-driven candidates.
- Optional training-normalized RMSE for validation-only method and variant selection.
- Dataset profiling, baseline comparisons and descriptive accuracy/runtime Pareto sets.
- Markdown recommendation reports, schema-2 result metadata and UI trajectory downloads.
- Equation JSON files for the CLI, restricted parsing and explicit column-order checks.
- Changed inputs hide previous UI recommendations; split isolation, unit scaling and packaged workflows are tested.

## 0.1.0 — initial library release candidate

- Public `benchmark`, `benchmark_data`, `evaluate`, candidate/model protocol and structured result objects.
- Chronological training/validation/test separation; validation-only variant selection and ranking.
- Numerical integration, parameter calibration, trajectory/data-driven candidates and optional neural candidates.
- State-based polynomial and neural acceleration corrections, with explicit initial-state transfer support.
- Safe numeric exports, separate fit/prediction/search costs, visible warnings and failed predictions.
- Saved parameter mismatch, noise/missing-physics, initial-state transfer and classical control studies.
- Installed CLI and bundled Streamlit client of the public API. Original UI retained in legacy_app.py.
- Custom expressions converted from a restricted AST instead of evaluating Python source; explicit invalid/non-finite input checks.
- Non-finite final scoring is recorded without reselecting a method using test outcomes.
- Packaging metadata, focused tests, GitHub CI/manual publishing workflows, three runnable tutorials and documentation.

This is a local release candidate. Public index publication and successful remote CI runs are not implied.
