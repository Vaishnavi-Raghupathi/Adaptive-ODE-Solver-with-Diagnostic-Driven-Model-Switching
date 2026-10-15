"""Generate tables and static figures directly from saved experiment results."""

from collections import Counter
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_ode.experiments.artifacts import write_json


def _method(case, name):
    return next(row for row in case["methods"] if row["method"] == name)


def summarize(config, results):
    summary = []
    for level in config["mismatch_levels"]:
        cases = [case for case in results if case["mismatch_level"] == level]
        for name in [row["method"] for row in cases[0]["methods"]]:
            rows = [_method(case, name) for case in cases]
            successful = [row for row in rows if row["status"] == "ok"]
            errors = [row["test_metrics"]["mse"] for row in successful]
            summary.append({
                "mismatch_level": level, "method": name,
                "successful_runs": len(successful), "failed_runs": len(rows) - len(successful),
                "mean_test_mse": float(np.mean(errors)) if errors else None,
                "std_test_mse": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
                "median_search_fit_seconds": float(np.median([row["search_fit_seconds"] for row in rows])),
                "median_prediction_seconds": float(np.median([row["prediction_seconds"] for row in rows])),
                "validation_selections": sum(row["selected_on_validation"] for row in rows),
            })
    return summary


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, config, results):
    output = Path(output)
    summary = summarize(config, results)
    write_json(output / "summary.json", summary)
    _write_csv(output / "summary.csv", summary)
    records = []
    for case in results:
        for row in case["methods"]:
            records.append({
                "case_id": case["case_id"], "mismatch_level": case["mismatch_level"],
                "seed": case["seed"], "method": row["method"], "variant": row["variant"],
                "status": row["status"], "message": row["message"],
                "selected_on_validation": row["selected_on_validation"],
                "eligible_for_selection": row["eligible_for_selection"],
                "validation_mse": row["validation_mse"],
                **{f"test_{metric}": (row["test_metrics"] or {}).get(metric) for metric in ["mse", "rmse", "mae"]},
                "fit_seconds": row["fit_seconds"], "prediction_seconds": row["prediction_seconds"],
                "search_fit_seconds": row["search_fit_seconds"],
                "search_prediction_seconds": row["search_prediction_seconds"],
            })
    _write_csv(output / "metrics.csv", records)
    failures = [
        {"case_id": case["case_id"], **row}
        for case in results for row in case["tuning_attempts"] if row["status"] != "ok"
    ]
    failures.extend({"case_id": case["case_id"], **row} for case in results
                    for row in case["methods"] if not row["eligible_for_selection"] and row["status"] != "ok")
    write_json(output / "failures.json", failures)
    convergence_warnings = sum(bool(row["details"].get("warnings"))
                               for case in results for row in case["tuning_attempts"]
                               if row["method"] == "rk4_residual_nn")
    nn_attempts = sum(row["method"] == "rk4_residual_nn"
                      for case in results for row in case["tuning_attempts"])
    wins = Counter(case["selected_method"] for case in results)
    lines = [
        "# Day 5: controlled oscillator parameter mismatch", "",
        "Generated from the saved case records; no table values are entered by hand.", "",
        "## Setup", "",
        f"- {len(results)} cases: {len(config['mismatch_levels'])} mismatch levels × {len(config['seeds'])} initialization seeds.",
        f"- True parameters: `{config['true_parameters']}`; fully misspecified parameters: `{config['wrong_parameters']}`.",
        f"- Initial state: `{config['initial_state']}`; time span: `{config['t_span']}`; {config['num_points']} points; no observation noise.",
        f"- Chronological split counts: `{results[0]['split_counts']}`.",
        f"- Split time ranges: `{results[0]['split_time_ranges']}`.",
        "- Both position and velocity are observed; the initial state is known exactly.",
        "- NN alpha and the recommended method are chosen on validation MSE against observations; final test values are accessed only after selection.",
        "- Models are not retrained on validation data. Test trajectories continue from the original initial state without resetting at split boundaries.",
        "- Analytical solution supplies truth. The true-equation DOP853 oracle is a numerical check and cannot be recommended.", "",
        "## Test error", "",
        "MSE averages position and velocity in their original units. ± is sample standard deviation over initialization seeds, not a confidence interval.", "",
        "| Mismatch | Method | Test MSE, mean ± SD | Successful / total | Validation selections |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        mse = "failed" if row["mean_test_mse"] is None else f"{row['mean_test_mse']:.3e}"
        if row["std_test_mse"] is not None:
            mse += f" ± {row['std_test_mse']:.2e}"
        lines.append(f"| {row['mismatch_level']:g} | {row['method']} | {mse} | {row['successful_runs']} / {len(config['seeds'])} | {row['validation_selections']} |")
    helpful = {}
    for level in config["mismatch_levels"]:
        count = 0
        cases = [case for case in results if case["mismatch_level"] == level]
        for case in cases:
            neural = _method(case, "rk4_residual_nn")
            baselines = [_method(case, name) for name in ["assumed_rk4", "fitted_dop853"]]
            if neural["status"] == "ok" and all(row["status"] == "ok" for row in baselines):
                error = neural["test_metrics"]["mse"]
                baseline_error = min(row["test_metrics"]["mse"] for row in baselines)
                count += error <= 0.9 * baseline_error and baseline_error - error >= 1e-8
        helpful[str(level)] = int(count)
    write_json(output / "interpretation.json", {
        "hybrid_helpful_seed_counts": helpful,
        "criterion": "At least 10% lower test MSE and absolute improvement >= 1e-8 versus both assumed RK4 and fitted DOP853, in >=80% of seeds at a level.",
        "validation_selections": dict(wins),
        "failed_attempts": len(failures),
        "nn_attempts_with_convergence_warning": convergence_warnings,
        "nn_attempts": nn_attempts,
    })
    lines += [
        "", "## Predeclared interpretation", "",
        "The hybrid counts as practically helpful at a mismatch level only if it lowers test MSE by at least 10%, with an absolute reduction of at least 1e-8, compared with both assumed RK4 and fitted DOP853 in at least 80% of seeds.",
        f"Qualifying seed counts by mismatch level: `{helpful}`.",
        f"Validation selections across all cases: `{dict(wins)}`.",
        f"Failed attempts: {len(failures)}. NN attempts emitting convergence warnings: {convergence_warnings}/{nn_attempts}. Warnings and loss curves are retained in each case JSON; a warning is not hidden as successful convergence.",
        "", "## Cost", "",
        "| Mismatch | Method | Median fit/search seconds | Median full-grid prediction seconds |",
        "|---|---|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['mismatch_level']:g} | {row['method']} | {row['median_search_fit_seconds']:.6f} | {row['median_prediction_seconds']:.6f} |")
    lines += [
        "", "The NN fit/search cost includes both alpha trials. Prediction includes a full baseline integration plus correction; it is not just a network forward pass. Additional prediction cost for the unselected alpha is in metrics.csv. Timings are one measurement per case on one machine with BLAS limited to one thread, not stable performance rankings; module import/first-use overhead can affect early timings. See provenance.json for environment and total run time.",
        "", "## Figures", "",
        "![Test error and training cost](accuracy_cost.png)", "",
        "![Prespecified first-seed trajectory and residuals at maximum mismatch](trajectory.png)",
        "", "## Limits", "",
        "- One synthetic linear oscillator and one exactly known initial state. No evidence yet for new initial conditions, noisy data, missing terms, or other system families.",
        "- Clean, fully observed data make parameter calibration unusually favorable. Its bounded single-start optimizer is not a universal identification algorithm.",
        "- The NN learns trajectory residuals from time and baseline states, not a corrected differential equation. This evaluation measures future continuation of that specific trajectory.",
        "- Repeated seeds vary NN initialization/minibatch order, not the deterministic reference trajectory. They are not independent physical experiments; no statistical significance claim is made.",
        "- The fixed NN budget and validation window may disadvantage extrapolation. No settings were changed in response to final test scores.",
        "- Lower error from a fit does not prove why it improved. Residual summaries are descriptive, not causal explanations.",
        "", "## Reproduce", "",
        "From the repository root, install requirements-research.lock and run:", "",
        "```bash",
        "python -m adaptive_ode.experiments.mismatch --config experiments/configs/day05_mismatch.json --output experiments/results/day05_reproduction",
        "```", "",
        "Use a new output directory. Each run preserves its resolved configuration, environment, exact source snapshot and hashes, per-case predictions, validation trials, and test results. Floating-point differences across library versions/hardware are possible; wall-clock times will vary.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    _plot_results(output, config, results, summary)


def _plot_results(output, config, results, summary):
    methods = [row["method"] for row in results[0]["methods"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for name in methods:
        rows = [row for row in summary if row["method"] == name and row["mean_test_mse"] is not None]
        axes[0].plot([row["mismatch_level"] for row in rows],
                     [max(row["mean_test_mse"], 1e-30) for row in rows], marker="o", label=name)
    axes[0].set(yscale="log", xlabel="Parameter mismatch fraction", ylabel="Mean future test MSE",
                title="Accuracy across fixed initialization seeds")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    rows = [row for row in summary if row["mismatch_level"] == max(config["mismatch_levels"])]
    axes[1].barh([row["method"] for row in rows],
                 [max(row["median_search_fit_seconds"], 1e-7) for row in rows])
    axes[1].set(xscale="log", xlabel="Median fit/search seconds (includes NN tuning)",
                title="Fit cost at maximum mismatch; prediction cost in report")
    fig.savefig(output / "accuracy_cost.png", dpi=160)
    plt.close(fig)

    case = next(case for case in results if case["mismatch_level"] == max(config["mismatch_levels"])
                and case["seed"] == config["seeds"][0])
    with np.load(output / "cases" / f"{case['case_id']}.npz", allow_pickle=False) as arrays:
        t, reference = arrays["t"], arrays["reference"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, layout="constrained")
        for dim, label in enumerate(["Position", "Velocity"]):
            axes[dim, 0].plot(t, reference[:, dim], color="black", linewidth=2, label="analytical reference")
            for name in ["assumed_rk4", "fitted_dop853", "rk4_residual_nn"]:
                row = _method(case, name)
                if row["status"] != "ok":
                    continue
                prediction = arrays[row["prediction_key"]]
                axes[dim, 0].plot(t, prediction[:, dim], linestyle="--", label=name, alpha=0.85)
                axes[dim, 1].plot(t, reference[:, dim] - prediction[:, dim], label=name)
            for ax in axes[dim]:
                ax.axvline(t[arrays["validation_indices"][0]], color="gray", linestyle=":")
                ax.axvspan(t[arrays["test_indices"][0]], t[-1], color="gray", alpha=0.12)
                ax.grid(alpha=0.2)
            axes[dim, 0].set_ylabel(label)
            axes[dim, 1].set_ylabel(f"{label} residual")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].set_xlabel("Time (dotted line: validation start; shading: final test)")
        axes[1, 1].set_xlabel("Time")
        fig.suptitle(f"Prespecified case: mismatch={case['mismatch_level']:g}, seed={case['seed']}")
        fig.savefig(output / "trajectory.png", dpi=160)
        plt.close(fig)
