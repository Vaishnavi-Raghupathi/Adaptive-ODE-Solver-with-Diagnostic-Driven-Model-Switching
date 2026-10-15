"""Frozen Days 6–7 robustness and initial-condition transfer experiment."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from adaptive_ode import BenchmarkConfig, evaluate
from adaptive_ode.evaluation.metrics import compute_metrics
from adaptive_ode.experiments.artifacts import create_run, finish_run, write_json
from adaptive_ode.experiments.oscillator import OscillatorParameters, analytical_trajectory
from adaptive_ode.methods import classical_candidate, oscillator_calibration_candidate, polynomial_correction_candidate
from adaptive_ode.ml_methods import acceleration_residual_candidate, trajectory_residual_candidate
from adaptive_ode.numerics import integrate, time_grid


def true_rhs(config, system):
    beta = config["missing_cubic_coefficient"] if system == "missing_cubic_force" else 0.0
    def rhs(time, state):
        del time
        x, v = state
        return np.array([v, -config["true_damping"]*v - config["true_stiffness"]*x - beta*x**3])
    return rhs


def reference_trajectory(config, system, initial):
    t = time_grid(np.linspace(*config["t_span"], config["num_points"]))
    rhs = true_rhs(config, system)
    if system == "linear_mismatch":
        reference = analytical_trajectory(t, initial, OscillatorParameters(config["true_damping"], config["true_stiffness"]))
        alternate = integrate(rhs, t, initial, rtol=1e-11, atol=1e-13, max_nfev=1000000)
        kind = "analytical versus DOP853"
    else:
        reference = integrate(rhs, t, initial, rtol=1e-11, atol=1e-13, max_nfev=1000000)
        alternate = integrate(rhs, t, initial, method="Radau", rtol=1e-11, atol=1e-13, max_nfev=1000000)
        kind = "DOP853 versus Radau"
    return reference, {"system": system, "initial_state": initial, "cross_check": kind,
                       "max_abs_difference": float(np.max(np.abs(reference-alternate))),
                       "mse_difference": compute_metrics(reference, alternate)["mse"]}


def candidates_for(config, seed):
    assumed = OscillatorParameters(config["assumed_damping"], config["assumed_stiffness"])
    budget = config["max_nfev"]
    candidates = [classical_candidate(assumed.rhs, method="RK4", name="assumed_rk4", max_nfev=budget),
                  classical_candidate(assumed.rhs, name="assumed_dop853", max_nfev=budget),
                  oscillator_calibration_candidate(assumed, max_nfev=budget)]
    candidates += [polynomial_correction_candidate(assumed, window=window, alpha=config["polynomial_alpha"], max_nfev=budget)
                   for window in config["smoothing_windows"]]
    for alpha in config["nn_alphas"]:
        candidates.append(trajectory_residual_candidate(assumed.rhs, seed=seed, alpha=alpha,
                          max_iter=config["nn_max_iter"], max_nfev=budget))
        candidates.append(acceleration_residual_candidate(assumed, seed=seed, alpha=alpha,
                          window=config["acceleration_nn_window"], max_iter=config["nn_max_iter"], max_nfev=budget))
    return candidates


def run_condition(config, system, noise, stride, seed, references):
    t = np.linspace(*config["t_span"], config["num_points"])
    reference = references[system][0]
    observed = reference + np.random.default_rng(seed).normal(0, noise, reference.shape)
    observed[0] = reference[0]  # Initial condition is known exactly by protocol.
    settings = BenchmarkConfig(
        train_fraction=config["train_fraction"], validation_fraction=config["validation_fraction"],
        num_points=len(t), seed=seed, nn_max_iter=config["nn_max_iter"],
        max_nfev=config["max_nfev"], train_stride=stride,
    )
    result = evaluate(t, observed, candidates_for(config, seed), config=settings, reference=reference,
                      name=f"{system}, noise={noise:g}, training stride={stride}")
    transfers, arrays = [], {}
    # Models/settings were frozen by evaluate. New targets are only used here.
    for index, initial in enumerate(config["held_out_initial_states"]):
        target = references[system][index+1]
        arrays[f"transfer_reference_{index}"] = target
        for method_index, row in enumerate(result.rows):
            record = {"method": row["method"], "initial_state_index": index, "initial_state": initial,
                      "selected_on_original_validation": row["selected"], "status": "not_supported",
                      "mse": None, "rmse": None, "mae": None, "prediction_seconds": 0.0, "message": ""}
            model = result.models.get(row["method"])
            if row["status"] != "ok":
                record.update(status="fit_failed", message=row["message"])
            elif model.transfer is not None:
                start = perf_counter()
                try:
                    prediction = model.predict_from_initial(t, initial)
                    record.update(status="ok", **compute_metrics(target, prediction))
                    arrays[f"transfer_{index}_method_{method_index}"] = prediction
                except Exception as error:
                    record.update(status="failed", message=f"{type(error).__name__}: {error}")
                record["prediction_seconds"] = perf_counter()-start
            transfers.append(record)
    return result, transfers, arrays


def _csv(path, records):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_summary(output, config, cases):
    groups = {}
    transfer_groups = {}
    records, transfer_records = [], []
    for case in cases:
        base = {key: case[key] for key in ["case_id", "system", "noise_std", "train_stride", "seed"]}
        for row in case["rows"]:
            records.append({**base, **{key: row[key] for key in ["method", "variant", "status", "validation_mse", "selected",
                "test_mse", "test_rmse", "test_mae", "fit_seconds", "prediction_seconds", "search_seconds", "message"]}})
            key = (case["system"], case["noise_std"], case["train_stride"], row["method"])
            groups.setdefault(key, []).append(row)
        for row in case["transfers"]:
            record = {**base, **row}
            record["initial_state"] = json.dumps(record["initial_state"])
            transfer_records.append(record)
            key = (case["system"], case["noise_std"], case["train_stride"], row["method"], row["initial_state_index"])
            transfer_groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        errors = [row["test_mse"] for row in rows if row["status"] == "ok"]
        summary.append(dict(zip(["system", "noise_std", "train_stride", "method"], key)) | {
            "mean_test_mse": float(np.mean(errors)) if errors else None,
            "std_test_mse": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
            "successful": len(errors), "failed": len(rows)-len(errors),
            "validation_selections": sum(row["selected"] for row in rows),
            "median_search_seconds": float(np.median([row["search_seconds"] for row in rows])),
        })
    transfer_summary = []
    for key, rows in transfer_groups.items():
        errors = [row["mse"] for row in rows if row["status"] == "ok"]
        transfer_summary.append(dict(zip(["system", "noise_std", "train_stride", "method", "initial_state_index"], key)) | {
            "mean_mse": float(np.mean(errors)) if errors else None,
            "successful": len(errors), "failed": sum(row["status"] in {"failed", "fit_failed"} for row in rows),
            "not_supported": sum(row["status"] == "not_supported" for row in rows),
        })
    _csv(output / "metrics.csv", records)
    _csv(output / "transfer_metrics.csv", transfer_records)
    _csv(output / "summary.csv", summary)
    _csv(output / "transfer_summary.csv", transfer_summary)
    write_json(output / "summary.json", summary)
    write_json(output / "transfer_summary.json", transfer_summary)
    wins = []
    transfer_wins = []
    for system in config["systems"]:
        for noise in config["noise_stds"]:
            for stride in config["train_strides"]:
                subset = [case for case in cases if case["system"] == system and case["noise_std"] == noise and case["train_stride"] == stride]
                for method in ["trajectory_residual_nn", "polynomial_correction", "acceleration_residual_nn"]:
                    count = 0
                    for case in subset:
                        baseline = next(row for row in case["rows"] if row["method"] == "calibrated_oscillator")
                        candidate = next(row for row in case["rows"] if row["method"] == method)
                        if candidate["status"] == baseline["status"] == "ok":
                            count += candidate["test_mse"] <= .9*baseline["test_mse"] and baseline["test_mse"]-candidate["test_mse"] >= 1e-6
                    wins.append({"system": system, "noise_std": noise, "train_stride": stride, "method": method,
                                 "qualifying_seeds": int(count), "total_seeds": len(subset)})
                    for initial_index in range(len(config["held_out_initial_states"])):
                        count = 0
                        for case in subset:
                            rows = [row for row in case["transfers"] if row["initial_state_index"] == initial_index]
                            baseline = next(row for row in rows if row["method"] == "calibrated_oscillator")
                            candidate = next(row for row in rows if row["method"] == method)
                            if candidate["status"] == baseline["status"] == "ok":
                                count += candidate["mse"] <= .9*baseline["mse"] and baseline["mse"]-candidate["mse"] >= 1e-6
                        transfer_wins.append({"system": system, "noise_std": noise, "train_stride": stride,
                                              "method": method, "initial_state_index": initial_index,
                                              "qualifying_seeds": int(count), "total_seeds": len(subset)})
    failed = [dict(case_id=case["case_id"], **row) for case in cases for row in case["trials"] if row["status"] != "ok"]
    warning_count = sum(bool(row["warnings"]) for case in cases for row in case["trials"])
    write_json(output / "failures.json", failed)
    write_json(output / "interpretation.json", {"future_wins_vs_calibration": wins, "transfer_wins_vs_calibration": transfer_wins,
               "trial_failures": len(failed), "trials_with_warnings": warning_count,
               "validation_selections": dict(Counter(case["selected_method"] for case in cases))})
    lines = ["# Days 6–7: robustness and initial-state transfer", "",
             f"Completed {len(cases)} cases. Configuration and protocol are preserved in source/.", "",
             "Training/validation use noisy observations; future test and transfer scores use clean synthetic truth. Variants and the recommended method are frozen before either final evaluation. No refitting on held-out initial states occurs.", "",
             "## Future test performance", "",
             "Mean ± sample SD over seeds; raw-state MSE. Failure counts are shown rather than silently excluded. Means describe successful predictions only.", "",
             "| System | Noise | Stride | Method | Test MSE mean ± SD | Successful/total | Validation selections |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for row in summary:
        value = "failed" if row["mean_test_mse"] is None else f"{row['mean_test_mse']:.3e}"
        if row["std_test_mse"] is not None:
            value += f" ± {row['std_test_mse']:.2e}"
        lines.append(f"| {row['system']} | {row['noise_std']:g} | {row['train_stride']} | {row['method']} | {value} | {row['successful']}/{row['successful']+row['failed']} | {row['validation_selections']} |")
    lines += ["", "## Practical improvement over calibration", "",
              "A condition qualifies only with >=10% and >=1e-6 lower MSE in >=80% of all seeds. Failures count as non-wins; this is not a significance test.", "",
              "| System | Noise | Stride | Method | Qualifying future seeds |",
              "|---|---:|---:|---|---:|"]
    for row in wins:
        lines.append(f"| {row['system']} | {row['noise_std']:g} | {row['train_stride']} | {row['method']} | {row['qualifying_seeds']}/{row['total_seeds']} |")
    lines += ["", "## Transfer and failures", "",
              "See transfer_summary.csv and transfer_metrics.csv for full-horizon errors at each unseen initial state. Transfer uses the original trajectory's validation-selected variant, with no new labels for fitting. The trajectory residual NN is explicitly not_supported.",
              f"Trial failures: {len(failed)}. Trials with recorded warnings: {warning_count}. Transfer failures remain in transfer_metrics.csv.", "",
              "## Limits", "",
              "- Two synthetic oscillator structures, fully observed position and velocity, known exact initial states; no general-purpose solver superiority claim.",
              "- The polynomial feature library includes the true missing cubic term. Its success is evidence for this favorable baseline, not automatic discovery of arbitrary physics.",
              "- Differentiating noisy/sparse velocities and choosing smoothing windows can dominate performance. No true derivatives were supplied to learners.",
              "- New initial states are three predetermined tests, not proof of unrestricted generalization. Means over successful cases can hide failures; inspect their counts.",
              "- Calibration can lose when its linear model class cannot represent a missing term; this does not establish that neural networks are the necessary remedy.",
              "- Cost is a single measurement on one machine with single-thread BLAS, not a statistically established speed ranking.", "",
              "![Prespecified missing-physics case](representative.png)", "",
              "Reproduce from the repository root using a new directory:", "", "```bash",
              "python -m adaptive_ode.experiments.robustness --config experiments/configs/day06_07_robustness.json --output experiments/results/robustness_reproduction", "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_study(config, output):
    if config.get("protocol_version") != 2:
        raise ValueError("Expected robustness protocol_version=2.")
    if any(name not in {"linear_mismatch", "missing_cubic_force"} for name in config["systems"]):
        raise ValueError("Unknown robustness system.")
    if any(not np.isfinite(noise) or noise < 0 for noise in config["noise_stds"]):
        raise ValueError("Noise standard deviations must be finite and nonnegative.")
    # Validate every split/budget before creating a run directory.
    for stride in config["train_strides"]:
        BenchmarkConfig(num_points=config["num_points"], train_fraction=config["train_fraction"],
                        validation_fraction=config["validation_fraction"], train_stride=stride,
                        nn_max_iter=config["nn_max_iter"], max_nfev=config["max_nfev"])
    with threadpool_limits(limits=1):
        output, metadata = create_run(output, config)
        start = perf_counter()
        references, checks = {}, []
        for system in config["systems"]:
            references[system] = []
            for initial in [config["training_initial_state"], *config["held_out_initial_states"]]:
                reference, check = reference_trajectory(config, system, initial)
                references[system].append(reference)
                checks.append(check)
        write_json(output / "reference_checks.json", checks)
        cases = []
        for system in config["systems"]:
            for noise in config["noise_stds"]:
                for stride in config["train_strides"]:
                    for seed in config["seeds"]:
                        case_id = f"{system}_noise{noise:g}_stride{stride}_seed{seed}"
                        result, transfers, arrays = run_condition(config, system, noise, stride, seed, references)
                        result.save(output / "cases" / case_id)
                        np.savez_compressed(output / "cases" / case_id / "transfer_predictions.npz", **arrays)
                        case = {"case_id": case_id, "system": system, "noise_std": noise, "train_stride": stride,
                                "seed": seed, "selected_method": result.selected_method, "rows": result.rows,
                                "trials": result.trials, "transfers": transfers}
                        write_json(output / "cases" / case_id / "transfer.json", transfers)
                        cases.append(case)
                        print(f"{case_id}: {result.selected_method}", flush=True)
                        if system == "missing_cubic_force" and noise == .02 and stride == 1 and seed == config["seeds"][0]:
                            import matplotlib
                            matplotlib.use("Agg")
                            import matplotlib.pyplot as plt
                            figure = result.plot()
                            figure.savefig(output / "representative.png", dpi=150)
                            plt.close(figure)
        write_json(output / "results.json", cases)
        write_summary(output, config, cases)
        finish_run(output, metadata, perf_counter()-start, cases=len(cases))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_study(json.loads(arguments.config.read_text()), arguments.output)


if __name__ == "__main__":
    main()
