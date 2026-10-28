# Contributing

Adaptive Solver Bench welcomes reproducible bug reports, method adapters, dataset examples and evaluation improvements. Open an issue describing the research use case, expected behavior, a minimal numeric example and dependency versions. Avoid uploading private datasets or credentials.

## Local checks

```bash
python -m pip install -e '.[research,demo,dev]'
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

Submit focused pull requests with an explanation of the resulting behavior and relevant validation. Public APIs should work from an installed wheel outside the checkout. Keep optional dependencies optional and declare new extras when needed.

## Scientific evaluation contract

- Fitting callbacks receive training data only. Do not capture held-out labels in closures or external state.
- Method/variant selection uses validation information and the declared accuracy/cost policy. Final test outcomes must not retune or replace a method.
- Treat independent experiments as the unit of splitting and uncertainty estimation. Do not claim overlapping windows are independent trials.
- Keep candidate failures visible. Never improve a score by dropping failed test experiments from its denominator.
- Differentiate observed-data prediction error, model-identification quality and numerical integration error. State what information each method receives.
- Report fitting, prediction and total search work separately. Timing repetitions do not create independent scientific observations.
- Label synthetic or semisynthetic demonstrations explicitly. Test fixtures and software tests do not establish research superiority.

For new methods, implement the public `Candidate`/`FittedModel` protocol and document dependencies, input assumptions, hyperparameters and computational budgets. Independent-experiment adapters need `fit_trajectories` and a transfer callback. See [the protocol](docs/independent_experiments.md). Include a meaningful test of the method's evaluation behavior, not just a test that repeats its implementation.

Saved research protocols and source snapshots are historical records. Add new studies with new output directories instead of overwriting prior results. Do not claim public-index publication, independent adoption or scientific generalization without evidence.
