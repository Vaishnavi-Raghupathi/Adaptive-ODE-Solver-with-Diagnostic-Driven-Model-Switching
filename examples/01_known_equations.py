"""Tutorial 1: numerical solvers versus parameter calibration."""

import argparse
from pathlib import Path

from adaptive_ode import BenchmarkConfig, benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark("mismatch", config=BenchmarkConfig(num_points=201))
    for row in result.ranking:
        print(row["method"], "validation MSE:", row["validation_mse"], "test MSE:", row["test_mse"])
    print("Selected using validation only:", result.selected_method)
    result.save(args.output)


if __name__ == "__main__":
    main()
