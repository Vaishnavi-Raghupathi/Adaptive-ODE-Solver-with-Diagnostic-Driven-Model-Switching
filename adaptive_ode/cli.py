"""Installed command line entry point; no repository-relative imports."""

import argparse
import csv
import importlib.util
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
    for command in [known, data]:
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--include-ml", action="store_true")
        command.add_argument("--seed", type=int, default=7)
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
                                 include_ml=arguments.include_ml, seed=arguments.seed)
        if arguments.command == "benchmark":
            result = benchmark(arguments.system, config=config)
        else:
            t, y = read_csv(arguments.csv_path, arguments.time_column, arguments.states)
            result = benchmark_data(t, y, config=config)
        result.save(arguments.output)
        print(f"Validation selected: {result.selected_method or 'no successful method'}")
        print(f"Results: {arguments.output.resolve()}")
        return 0 if result.selected_method else 1
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
