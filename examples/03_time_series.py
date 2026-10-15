"""Tutorial 3: observations-only comparison, using a synthetic CSV example."""

import argparse
from pathlib import Path

from adaptive_ode import benchmark_data
from adaptive_ode.cli import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("data") / "oscillator.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    t, y = read_csv(args.csv)
    result = benchmark_data(t, y)
    print("Selected on validation observations:", result.selected_method)
    print("Test targets are observations, not independently verified ground truth.")
    for row in result.ranking:
        print(row["method"], row["status"], row["test_mse"])
    result.save(args.output)


if __name__ == "__main__":
    main()
