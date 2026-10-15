# Research experiments and release verification

This directory contains controlled evaluations and their evidence. The reusable library is in adaptive_ode; the current Streamlit client calls its public API. Historical source snapshots remain immutable.

## Deliverables

| Milestone | Artifact |
|---|---|
| Day 1 | `docs/day01_audit.md`, `results/day01_baseline/`, exact research dependency lock. |
| Day 2 | `docs/experiment_protocol.md`, `configs/day05_mismatch.json`. |
| Day 3 | `adaptive_ode/experiments/oscillator.py`, focused numerical verification tests. |
| Day 4 | `adaptive_ode/experiments/mismatch.py`, run provenance and recoverable source snapshots. |
| Day 5 | `results/day05_mismatch/report.md`, CSV/JSON metrics, predictions, figures, failures and tuning records. |
| Days 6–7 | `results/day06_07_robustness/report.md`, noise/data-density conditions, missing physics, three held-out initial states, full failures and transfer arrays. |
| Day 8 | `results/day08_classical/report.md`, four systems with bounded solver attempts and cross-checked references. |
| Release | `results/release_verification/`, installed-wheel checks, test XML and distribution hashes; `release_smoke/` contains tutorial/CLI exports. |

## Environment and commands

Run from the repository root. Python 3.13 on macOS ARM64 is the recorded environment. The research lock is separate from the original app requirements; Streamlit, torch and torchdiffeq are unnecessary for this study.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.lock
python -m pip check
python -m pytest tests -q

# Capture the original two-way-split app benchmark:
python -m adaptive_ode.experiments.capture_baseline --output experiments/results/day01_reproduction

# Run the frozen three-way-split experiment:
python -m adaptive_ode.experiments.mismatch --config experiments/configs/day05_mismatch.json --output experiments/results/day05_reproduction
```

Every command refuses to overwrite an existing run directory. An interrupted run keeps its completed case artifacts and has `status: running` in provenance, not a false completion marker; rerun into a fresh directory. There is no resume feature in this milestone. Dependencies must resolve on your platform; numerical tolerances allow floating-point differences, and runtimes will vary.

## Reading a completed run

- `report.md`: setup, findings, interpretation rule, costs, plots, limitations.
- `config.json`: resolved configuration; original production config lives in `configs/`.
- `metrics.csv`: one row per case and method, including selection, error and timing.
- `summary.csv` / `summary.json`: aggregates across initialization seeds.
- `results.json`, `cases/*.json`: all results, parameter fits, NN settings/loss curves/warnings, validation scores and selections.
- `cases/*.npz`: reference, observations, exact split indices and all trial predictions; read with `numpy.load(..., allow_pickle=False)`.
- `failures.json`: unsuccessful trials, including messages; an empty list means no failed attempts, not that every NN converged.
- `interpretation.json`: predeclared practical-improvement checks and convergence-warning counts.
- `provenance.json`: timestamps, completion, Git revision/status, platform, Python, thread limits, configuration hash and total elapsed time.
- `environment.txt`: exact installed distributions.
- `source/`, `source_manifest.json`: the exact code, tests, config, protocol and dependency lock used for that run, with SHA-256 hashes. This preserves uncommitted code as well as the base Git revision.

To reproduce an old run after the repository changes, copy that run's `source/` into a new working directory, create a fresh environment using its lock, and execute the recorded configuration there. The source snapshot excludes Streamlit because the research runner does not need it.

## Interpretation boundary

The legacy baseline and Day 5 study have different splits, sample counts, and baselines. Their errors are not a controlled before/after comparison. Day 5 studies future continuation of one synthetic trajectory with parameter errors, not missing terms or noisy data. Repeated seeds measure NN training variability, not independent physical experiments. Read the audit before reusing app PINN or legacy Neural ODE code in research comparisons.

The completed Days 6–14 local milestones and external publication/remote-CI boundaries are described in docs/days06_14_summary.md. For the new API and later studies, install `.[research,demo,dev]`; do not treat the original Day 5 lock as the current library environment. Every study preserves its own environment and source snapshot.
