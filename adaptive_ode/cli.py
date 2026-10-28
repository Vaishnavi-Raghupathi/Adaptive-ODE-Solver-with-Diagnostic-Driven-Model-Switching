"""Installed command line entry point; no repository-relative imports."""

import argparse
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from adaptive_ode import BenchmarkConfig, benchmark, benchmark_data
from adaptive_ode.systems import SYSTEMS


def read_csv(path, time_column="time", state_columns=None):
    """Read explicit numeric columns without requiring pandas."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique.")
        if time_column not in header:
            raise ValueError(f"Missing time column {time_column!r}.")
        states = state_columns or [column for column in header if column != time_column]
        if not states or time_column in states or len(states) != len(set(states)) or any(column not in header for column in states):
            raise ValueError("Select unique state columns distinct from the time column.")
        rows = list(reader)
        if len(rows) < 12:
            raise ValueError("CSV requires at least 12 data rows.")
        t = np.array([float(row[time_column]) for row in rows])
        y = np.array([[float(row[column]) for column in states] for row in rows])
    return t, y


def read_trajectory_manifest(path):
    """Load explicit experiment roles; paths are relative to the manifest."""
    from adaptive_ode import Trajectory
    path = Path(path)
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict) or set(spec) - {"train", "validation", "test", "time_column", "state_columns", "equations"}:
        raise ValueError("Manifest must contain train/validation/test lists and optional time_column, state_columns, equations.")
    datasets, source_files = {}, {}
    for role in ("train", "validation", "test"):
        entries = spec.get(role)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest needs a nonempty {role} list.")
        datasets[role], source_files[role] = [], []
        for entry in entries:
            if isinstance(entry, str):
                entry = {"path": entry}
            if not isinstance(entry, dict) or set(entry) - {"path", "name"} or not isinstance(entry.get("path"), str):
                raise ValueError("Each manifest entry must be a CSV path or an object with path and optional name.")
            source = path.parent / entry["path"]
            times, states = read_csv(source, spec.get("time_column", "time"), spec.get("state_columns"))
            name = entry.get("name", source.stem)
            datasets[role].append(Trajectory(times, states, name))
            source_files[role].append({"name": name, "path": entry["path"], "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    rhs, equations = None, None
    if spec.get("equations") is not None:
        if not isinstance(spec["equations"], str):
            raise ValueError("Manifest equations must be a JSON file path.")
        if importlib.util.find_spec("sympy") is None:
            raise ValueError("Equation files require: pip install 'adaptive-solver-bench[custom]'")
        from adaptive_ode.custom_system import build_observed_system
        equations = json.loads((path.parent / spec["equations"]).read_text())
        initial = datasets["train"][0]
        rhs = build_observed_system(equations, initial.t, initial.states).rhs
    return datasets, rhs, {"manifest": spec, "source_files": source_files, "equation_spec": equations}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="adaptive-solver", description="Compare dynamical-system methods using validation-only selection.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    known = subparsers.add_parser("benchmark", help="Run a built-in known-equation benchmark.")
    known.add_argument("--system", choices=list(SYSTEMS), default="mismatch")
    known.add_argument("--points", type=int, default=301)
    data = subparsers.add_parser("data", help="Evaluate candidates on numeric CSV observations.")
    data.add_argument("csv_path", type=Path)
    data.add_argument("--time-column", default="time")
    data.add_argument("--states", nargs="+")
    data.add_argument("--equations", type=Path, help="JSON equations in selected state-column order; requires [custom].")
    multiple = subparsers.add_parser("trajectories", help="Independent train/validation/test experiments from a JSON manifest.")
    multiple.add_argument("manifest", type=Path)
    for command in [known, data, multiple]:
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--include-ml", action="store_true")
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--metric", choices=["mse", "normalized_rmse"], default="mse",
                             help="Validation selection metric; normalized_rmse uses training-only state scales.")
        command.add_argument("--policy", choices=["min_error", "fastest_within_target"], default="min_error")
        command.add_argument("--accuracy-target", type=float, help="Maximum validation score; required for fastest_within_target.")
        command.add_argument("--cost", choices=["prediction_seconds", "fit_plus_prediction_seconds"], default="prediction_seconds")
        command.add_argument("--timing-repeats", type=int, default=1)
        command.add_argument("--solver-sweep", action="store_true", help="Compare preset tolerance and RK4 step-size variants.")
    multiple.set_defaults(metric="normalized_rmse", timing_repeats=3)
    demo = subparsers.add_parser("demo", help="Launch the optional bundled Streamlit demo.")
    demo.add_argument("--port", type=int, default=8501)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "demo":
            if importlib.util.find_spec("streamlit") is None:
                parser.error("Install the demo extra: pip install 'adaptive-solver-bench[demo]'")
            if not 1 <= arguments.port <= 65535:
                parser.error("port must be between 1 and 65535")
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("demo.py")),
                                    "--server.port", str(arguments.port)])
        if arguments.output.exists():
            raise FileExistsError(f"Output directory already exists: {arguments.output}")
        if arguments.include_ml and importlib.util.find_spec("sklearn") is None:
            raise ValueError("Install ML candidates with: pip install 'adaptive-solver-bench[ml]'")
        config = BenchmarkConfig(num_points=getattr(arguments, "points", 301),
                                 include_ml=arguments.include_ml, seed=arguments.seed,
                                 selection_metric=arguments.metric, selection_policy=arguments.policy,
                                 accuracy_target=arguments.accuracy_target, selection_cost=arguments.cost,
                                 timing_repeats=arguments.timing_repeats, solver_sweep=arguments.solver_sweep)
        if arguments.command == "benchmark":
            result = benchmark(arguments.system, config=config)
        elif arguments.command == "trajectories":
            from adaptive_ode import benchmark_trajectories
            datasets, rhs, context = read_trajectory_manifest(arguments.manifest)
            result = benchmark_trajectories(**datasets, rhs=rhs, config=config)
            result.context.update(context)
        else:
            t, y = read_csv(arguments.csv_path, arguments.time_column, arguments.states)
            rhs = None
            spec = None
            if arguments.equations:
                if importlib.util.find_spec("sympy") is None:
                    raise ValueError("Equation files require: pip install 'adaptive-solver-bench[custom]'")
                from adaptive_ode.custom_system import build_observed_system
                spec = json.loads(arguments.equations.read_text())
                system = build_observed_system(spec, t, y)
                rhs = system.rhs
            result = benchmark_data(t, y, config=config, rhs=rhs)
            result.context.update({"time_column": arguments.time_column, "state_columns": arguments.states,
                                   "source_sha256": hashlib.sha256(arguments.csv_path.read_bytes()).hexdigest(),
                                   "equation_spec": spec})
        result.save(arguments.output)
        print(f"Validation selected ({config.selection_metric}): {result.selected_method or 'no method meets the selection rule'}")
        print(f"Comparison: {result.assessment.get('comparison_scope', result.assessment.get('protocol'))}")
        for warning in result.assessment.get("warnings", []):
            print(f"Note: {warning}")
        print(f"Results: {arguments.output.resolve()}")
        return 0 if result.selected_method else 1
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
