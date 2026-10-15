"""Run Day 8 controls using the public API, preserving failures and references."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from adaptive_ode import BenchmarkConfig, benchmark
from adaptive_ode.experiments.artifacts import create_run, finish_run, write_json
from adaptive_ode.numerics import integrate
from adaptive_ode.systems import get_system


def run_study(config, output):
    settings = BenchmarkConfig(num_points=config["num_points"], train_fraction=config["train_fraction"],
                               validation_fraction=config["validation_fraction"], max_nfev=config["max_nfev"])
    with threadpool_limits(limits=1):
        output, metadata = create_run(output, config)
        start = perf_counter()
        records, checks = [], []
        for key in config["systems"]:
            for repeat in range(config["timing_repeats"]):
                result = benchmark(key, config=settings)
                result.save(output / "cases" / f"{key}_repeat{repeat}")
                if repeat == 0:
                    system = get_system(key)
                    alternate_method = {"lorenz": "Radau", "vanderpol": "DOP853", "robertson": "BDF", "mismatch": "DOP853"}[key]
                    alternate = integrate(system.rhs, result.t, system.y0, method=alternate_method,
                                          rtol=1e-11, atol=1e-13, max_nfev=1000000)
                    checks.append({"system": key, "reference_kind": result.reference_kind,
                                   "alternate_method": alternate_method,
                                   "max_abs_difference": float(np.max(np.abs(alternate-result.target)))})
                for row in result.rows:
                    records.append({"system": key, "repeat": repeat, **{name: row[name] for name in [
                        "method", "status", "validation_mse", "test_mse", "selected", "search_seconds", "message"]}})
                print(f"{key} repeat {repeat}: {result.selected_method}", flush=True)
        write_json(output / "reference_checks.json", checks)
        write_json(output / "results.json", records)
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        lines = ["# Day 8: classical benchmark controls", "",
                 "Deterministic repeats measure timing variation only. Candidate failures remain visible; numerical reference disagreements are in reference_checks.json.", "",
                 "| System | Repeat | Method | Status | Validation MSE | Future test MSE | Selected |",
                 "|---|---:|---|---|---:|---:|---|"]
        for row in records:
            validation = "—" if row["validation_mse"] is None else f"{row['validation_mse']:.3e}"
            test = "—" if row["test_mse"] is None else f"{row['test_mse']:.3e}"
            lines.append(f"| {row['system']} | {row['repeat']} | {row['method']} | {row['status']} | {validation} | {test} | {row['selected']} |")
        lines += ["", "Metrics average state dimensions in raw units, so do not compare absolute MSE between systems. Selection is validation-based and ignores runtime; timing is reported separately. Read docs/classical_protocol.md for scope and budgets."]
        (output / "report.md").write_text("\n".join(lines) + "\n")
        finish_run(output, metadata, perf_counter()-start,
                   failed_predictions=sum(row["status"] != "ok" for row in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_study(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
