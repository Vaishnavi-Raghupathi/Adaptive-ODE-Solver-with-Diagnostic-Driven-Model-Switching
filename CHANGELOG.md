# Changelog

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
