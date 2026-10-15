# Build, verify, and publish

The first deliverable is an installable **0.1.0 alpha** library and built wheel/source distribution. It is not a claim of production readiness, public PyPI availability, or a new numerical integration algorithm.

## Local installation and build

```bash
python -m pip install -e '.[research,demo,dev]'
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

`pyproject.toml` uses the standard project metadata/build-system tables, Apache-2.0 license metadata, and optional dependency extras. Version comes from `adaptive_ode.__version__`. See the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

Install a built wheel from any directory with an absolute file path:

```bash
python -m pip install /absolute/path/to/dist/adaptive_solver_bench-0.1.0-py3-none-any.whl
adaptive-solver benchmark --system mismatch --output runs/first
```

The base install uses only NumPy and SciPy. Extras: `ml`, `plots`, `custom`, `demo`, `research`, `dev`. The wheel includes package Python code and the bundled demo, not large archived benchmark artifacts. The source distribution includes tutorials/tests/configs/docs; study outputs stay in the repository. Results exports contain numeric NPZ/JSON/CSV, not pickle files or serialized fitted models.

## CI and release automation

`.github/workflows/ci.yml` runs tests on Python 3.11–3.13 Linux and Python 3.13 macOS, builds distributions, checks metadata, installs a wheel in a clean environment, and runs the three tutorials outside the checkout. This is the configured matrix, not proof that those remote jobs have executed. Local verification is recorded separately in docs/days06_14_summary.md. See [GitHub's Python CI guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

`.github/workflows/release.yml` is manually dispatched. Its default `artifacts` target only builds distributions after tests. `testpypi` and `pypi` targets require a Git tag matching `v` plus the package version and a separately configured trusted publisher. Simply pushing code does not publish a package. Neither workflow has been pushed or remotely executed by this local milestone.

## Public PyPI release prerequisites

1. Review and push the source, run the configured CI matrix, and inspect the generated distributions.
2. Recheck the proposed distribution name `adaptive-solver-bench`; a project-name lookup returned HTTP 404 during this work, which neither reserves nor guarantees that name.
3. Configure pending/trusted publishers on TestPyPI and PyPI for this repository, workflow `release.yml`, and the corresponding GitHub environments `testpypi`/`pypi`.
4. Tag the reviewed version, manually dispatch a TestPyPI release on that tag, and verify installation. TestPyPI dependency availability differs; use PyPI for ordinary dependencies as described in the packaging guide.
5. Manually dispatch the PyPI target only when ready to publish. Published version files cannot be replaced; version subsequent fixes.

Follow the official [trusted-publishing workflow guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/). No account setup, tag push, GitHub release, or package publication is performed as part of the local deliverable.

## Maintenance policy for the alpha

- Keep saved research runs immutable; a new scientific claim requires a new protocol, not edits to an old final test set.
- Run meaningful tests and installed-wheel examples before releases.
- Treat API/schema changes as release-note items; the alpha interface may evolve.
- Preserve unsuccessful methods and numerical failures in reports.
- Do not advertise optional experimental PINN/legacy Neural ODE code as validated features.
