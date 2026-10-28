"""Independent experiments, solver settings, and accuracy-constrained selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_ode import BenchmarkConfig, Trajectory, benchmark_trajectories
from adaptive_ode.oscillator import OscillatorParameters, analytical_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--without-equations", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory.")
    parameters = OscillatorParameters(.1, 4)
    rng = np.random.default_rng(3107)
    datasets = {}
    for role in ("train", "validation", "test"):
        datasets[role] = []
        for index in range(3):
            t = np.linspace(0, 6, 81)
            initial = rng.uniform([.6, -.4], [2.2, .4])
            states = analytical_trajectory(t, initial, parameters) + rng.normal(0, .002, (len(t), 2))
            datasets[role].append(Trajectory(t, states, f"{role}-{index+1}"))
    config = BenchmarkConfig(selection_metric="normalized_rmse", selection_policy="fastest_within_target",
                             accuracy_target=.05, timing_repeats=args.timing_repeats,
                             solver_sweep=not args.without_equations)
    result = benchmark_trajectories(**datasets, rhs=None if args.without_equations else parameters.rhs, config=config)
    equations = {"variables": ["position", "velocity"], "parameters": {"c": .1, "k": 4},
                 "derivatives": ["velocity", "-c*velocity-k*position"]}
    result.context.update({"data_kind": "synthetic independent oscillator experiments", "generator_seed": 3107,
                           "measurement_noise_sd": .002, "equation_spec": None if args.without_equations else equations})
    result.save(args.output)
    (args.output / "inputs").mkdir()
    manifest = {"time_column": "time", "state_columns": ["position", "velocity"]}
    for role, records in datasets.items():
        manifest[role] = []
        for record in records:
            relative = f"inputs/{record.name}.csv"
            np.savetxt(args.output / relative, np.column_stack([record.t, record.states]),
                       delimiter=",", header="time,position,velocity", comments="")
            manifest[role].append(relative)
    if not args.without_equations:
        (args.output / "equations.json").write_text(json.dumps(equations, indent=2) + "\n")
        manifest["equations"] = "equations.json"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validation selected: {result.selected_method or 'no qualifying method'}")
    print(f"Report: {(args.output / 'report.md').resolve()}")
    return 0 if result.selected_method else 1


if __name__ == "__main__":
    raise SystemExit(main())
