"""Controlled parameter-mismatch study; see docs/experiment_protocol.md.

Run from the repository root with:
python -m adaptive_ode.experiments.mismatch --config PATH --output NEW_DIRECTORY
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from adaptive_ode.evaluation.metrics import compute_metrics
from adaptive_ode.experiments.artifacts import create_run, finish_run, write_json
from adaptive_ode.experiments.oscillator import (
    OscillatorParameters, analytical_trajectory, fit_parameters,
    integrate_oscillator, validate_initial_state,
)
from adaptive_ode.solvers.candidates import solve_rk4


def chronological_split(n_points, train_fraction, validation_fraction):
    """Return three exhaustive, non-overlapping blocks in temporal order."""
    if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 12:
        raise ValueError("At least 12 integer samples are required.")
    fractions = np.array([train_fraction, validation_fraction], dtype=float)
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0) or fractions.sum() >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    train_end = int(n_points * train_fraction)
    validation_end = train_end + int(n_points * validation_fraction)
    if min(train_end, validation_end - train_end, n_points - validation_end) < 2:
        raise ValueError("Each split requires at least two points.")
    return {
        "train": np.arange(train_end),
        "validation": np.arange(train_end, validation_end),
        "test": np.arange(validation_end, n_points),
    }


def validate_config(config):
    expected = {
        "study_id", "protocol_version", "t_span", "num_points", "initial_state",
        "true_parameters", "wrong_parameters", "mismatch_levels", "seeds",
        "noise_std", "train_fraction", "validation_fraction", "nn",
    }
    if set(config) != expected:
        raise ValueError(f"Expected config keys: {sorted(expected)}")
    if config["protocol_version"] != 1:
        raise ValueError("Unsupported protocol_version.")
    if not isinstance(config["study_id"], str) or not config["study_id"].strip():
        raise ValueError("study_id must be a non-empty string.")
    chronological_split(config["num_points"], config["train_fraction"], config["validation_fraction"])
    span = np.asarray(config["t_span"], dtype=float)
    if span.shape != (2,) or not np.all(np.isfinite(span)) or span[1] <= span[0]:
        raise ValueError("t_span must contain two increasing finite values.")
    validate_initial_state(config["initial_state"])
    truth = OscillatorParameters(**config["true_parameters"])
    wrong = OscillatorParameters(**config["wrong_parameters"])
    if truth.stiffness <= (truth.damping / 2)**2:
        raise ValueError("The analytical reference requires underdamped true dynamics.")
    # The single-start parameter fit uses fixed, declared bounds.
    if not (0 <= wrong.damping <= 2 and 0.5 <= wrong.stiffness <= 8):
        raise ValueError("wrong_parameters must lie within the parameter-fit bounds.")
    if not (0 <= truth.damping <= 2 and 0.5 <= truth.stiffness <= 8):
        raise ValueError("true_parameters must lie within the parameter-fit bounds.")
    levels = config["mismatch_levels"]
    if not levels or len(set(levels)) != len(levels) or any(
        not np.isfinite(level) or not 0 <= level <= 1 for level in levels
    ):
        raise ValueError("mismatch_levels must be unique finite numbers in [0, 1].")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError("seeds must be unique nonnegative 32-bit integers.")
    if not np.isfinite(config["noise_std"]) or config["noise_std"] != 0:
        raise ValueError("Protocol v1 is the noiseless Day 5 study; noise belongs to a new protocol.")
    nn = config["nn"]
    if set(nn) != {"hidden_layer_sizes", "activation", "solver", "learning_rate_init", "max_iter", "alphas"}:
        raise ValueError("Unexpected nn configuration keys.")
    if nn["activation"] != "tanh" or nn["solver"] != "adam":
        raise ValueError("Protocol v1 uses tanh and adam.")
    if not nn["hidden_layer_sizes"] or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in nn["hidden_layer_sizes"]
    ):
        raise ValueError("Hidden layer sizes must be positive integers.")
    if not isinstance(nn["max_iter"], int) or isinstance(nn["max_iter"], bool) or nn["max_iter"] < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(nn["learning_rate_init"]) or nn["learning_rate_init"] <= 0:
        raise ValueError("learning_rate_init must be positive and finite.")
    if not nn["alphas"] or len(set(nn["alphas"])) != len(nn["alphas"]) or any(
        not np.isfinite(alpha) or alpha < 0 for alpha in nn["alphas"]
    ):
        raise ValueError("alphas must be unique, nonnegative and finite.")


def _fit_residual(t_train, y_train, y0, assumed, settings, alpha, seed):
    baseline = solve_rk4(assumed.rhs, t_train, y0)
    features = np.column_stack([t_train, baseline])
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=tuple(settings["hidden_layer_sizes"]),
            activation=settings["activation"], solver=settings["solver"],
            alpha=alpha, learning_rate_init=settings["learning_rate_init"],
            max_iter=settings["max_iter"], random_state=seed,
            # Disable internal early stopping / its implicit random holdout.
            early_stopping=False, shuffle=True, tol=1e-4, n_iter_no_change=10,
            batch_size="auto", beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, y_train - baseline)

    def predict(t):
        baseline_all = solve_rk4(assumed.rhs, t, y0)
        return baseline_all + model.predict(np.column_stack([t, baseline_all]))

    learner = model.steps[-1][1]
    return predict, {
        "alpha": alpha, "n_iter": int(learner.n_iter_), "loss": float(learner.loss_),
        "warnings": [str(item.message) for item in caught],
        "loss_curve": list(map(float, learner.loss_curve_)),
    }


def _attempt(name, variant, fitter, t):
    row = {
        "method": name, "variant": variant, "status": "failed", "message": "",
        "fit_seconds": 0.0, "prediction_seconds": 0.0,
        "validation_mse": None, "details": {},
    }
    start = perf_counter()
    try:
        predict, details = fitter()
        row["fit_seconds"] = perf_counter() - start
        row["details"] = details
        start = perf_counter()
        prediction = np.asarray(predict(t), dtype=float)
        row["prediction_seconds"] = perf_counter() - start
        if prediction.shape != (len(t), 2) or not np.all(np.isfinite(prediction)):
            raise RuntimeError("Prediction must be finite with shape (n, 2).")
        row["status"] = "ok"
        return row, prediction
    except Exception as error:
        # Failed attempts remain in the artifacts and their time is retained.
        if row["fit_seconds"] == 0:
            row["fit_seconds"] = perf_counter() - start
        else:
            row["prediction_seconds"] = perf_counter() - start
        row["message"] = f"{type(error).__name__}: {error}"
        return row, None


def choose_on_validation(rows):
    """Choose using validation MSE alone; input contains no test scores."""
    eligible = [row for row in rows if row["status"] == "ok"
                and row["validation_mse"] is not None
                and np.isfinite(row["validation_mse"])]
    return min(eligible, key=lambda row: row["validation_mse"]) if eligible else None


def fit_and_select(t, y_observed, split, y0, assumed, config, seed):
    """Consume train/validation labels only; never inspect test observations.

    Full-grid predictions use times and a known initial state, not future state
    values. Trial predictions are cached before any final test labels are read.
    """
    t_train = t[split["train"]]
    y_train = y_observed[split["train"]].copy()
    attempts = []
    predictions = {}

    def add(name, variant, fitter):
        row, prediction = _attempt(name, variant, fitter, t)
        row["prediction_key"] = f"prediction_{len(attempts):02d}"
        if prediction is not None:
            row["validation_mse"] = compute_metrics(
                y_observed[split["validation"]], prediction[split["validation"]],
            )["mse"]
            predictions[row["prediction_key"]] = prediction
        attempts.append(row)

    add("assumed_rk4", "fixed_grid", lambda: (
        lambda times: solve_rk4(assumed.rhs, times, y0), asdict(assumed),
    ))
    for name, rtol, atol in [("assumed_dop853", 1e-6, 1e-8), ("assumed_dop853_tight", 1e-10, 1e-12)]:
        add(name, "fixed", lambda rtol=rtol, atol=atol: (
            lambda times: integrate_oscillator(times, y0, assumed, rtol=rtol, atol=atol),
            {**asdict(assumed), "rtol": rtol, "atol": atol},
        ))

    def fit_physics():
        fitted, details = fit_parameters(t_train, y_train, y0, assumed)
        return (
            lambda times: integrate_oscillator(times, y0, fitted, rtol=1e-9, atol=1e-11),
            {**details, "parameters": asdict(fitted)},
        )

    add("fitted_dop853", "single_start", fit_physics)
    for alpha in config["nn"]["alphas"]:
        add("rk4_residual_nn", f"alpha={alpha:g}", lambda alpha=alpha: _fit_residual(
            t_train, y_train, y0, assumed, config["nn"], alpha, seed,
        ))

    # First select an NN regularization setting, then compare method families.
    selected_rows = []
    for name in dict.fromkeys(row["method"] for row in attempts):
        family = [row for row in attempts if row["method"] == name]
        best = choose_on_validation(family)
        row = dict(best if best is not None else family[0])
        row["search_fit_seconds"] = sum(item["fit_seconds"] for item in family)
        row["search_prediction_seconds"] = sum(item["prediction_seconds"] for item in family)
        selected_rows.append(row)
    selected = choose_on_validation(selected_rows)
    return attempts, selected_rows, predictions, selected["method"] if selected else None


def evaluate_test(rows, predictions, reference, observed, test_indices):
    """Post-selection reporting; cannot alter the chosen method or NN setting."""
    scored = []
    for original in rows:
        row = dict(original)
        row["test_metrics"] = None
        row["observed_test_metrics"] = None
        row["test_residuals"] = None
        if row["status"] == "ok":
            prediction = predictions[row["prediction_key"]][test_indices]
            target = reference[test_indices]
            row["test_metrics"] = compute_metrics(target, prediction)
            row["observed_test_metrics"] = compute_metrics(observed[test_indices], prediction)
            residuals = target - prediction
            # Descriptive per-state diagnostics, not hypothesis-test evidence.
            lag1 = []
            for signal in residuals.T:
                lag1.append(float(np.corrcoef(signal[:-1], signal[1:])[0, 1])
                            if np.std(signal[:-1]) > 1e-12 and np.std(signal[1:]) > 1e-12 else None)
            row["test_residuals"] = {
                "mean_by_state": residuals.mean(axis=0).tolist(),
                "std_by_state": residuals.std(axis=0).tolist(),
                "max_abs_by_state": np.abs(residuals).max(axis=0).tolist(),
                "lag1_correlation_by_state": lag1,
            }
        scored.append(row)
    return scored


def run_case(config, level, seed):
    t = np.linspace(*config["t_span"], config["num_points"])
    y0 = np.asarray(config["initial_state"], dtype=float)
    truth = OscillatorParameters(**config["true_parameters"])
    wrong = OscillatorParameters(**config["wrong_parameters"])
    assumed = OscillatorParameters(
        truth.damping + level * (wrong.damping - truth.damping),
        truth.stiffness + level * (wrong.stiffness - truth.stiffness),
    )
    reference = analytical_trajectory(t, y0, truth)
    observed = reference.copy()  # Protocol v1 is deliberately noise-free.
    split = chronological_split(len(t), config["train_fraction"], config["validation_fraction"])
    attempts, rows, predictions, selected = fit_and_select(
        t, observed, split, y0, assumed, config, seed,
    )
    # Selection is already frozen. Only now compute final test metrics.
    rows = evaluate_test(rows, predictions, reference, observed, split["test"])
    for row in rows:
        row["selected_on_validation"] = row["method"] == selected
        row["eligible_for_selection"] = True

    oracle, prediction = _attempt("oracle_dop853", "true_equations", lambda: (
        lambda times: integrate_oscillator(times, y0, truth, rtol=1e-10, atol=1e-12),
        {**asdict(truth), "rtol": 1e-10, "atol": 1e-12},
    ), t)
    oracle["prediction_key"] = "prediction_oracle"
    oracle["search_fit_seconds"] = oracle["fit_seconds"]
    oracle["search_prediction_seconds"] = oracle["prediction_seconds"]
    if prediction is not None:
        predictions[oracle["prediction_key"]] = prediction
    oracle = evaluate_test([oracle], predictions, reference, observed, split["test"])[0]
    oracle.update(selected_on_validation=False, eligible_for_selection=False)
    rows.append(oracle)

    result = {
        "mismatch_level": level, "seed": seed,
        "assumed_parameters": asdict(assumed),
        "selected_method": selected,
        "split_counts": {key: len(indices) for key, indices in split.items()},
        "split_time_ranges": {key: [float(t[indices[0]]), float(t[indices[-1]])] for key, indices in split.items()},
        "methods": rows, "tuning_attempts": attempts,
    }
    arrays = {"t": t, "reference": reference, "observed": observed, **predictions}
    arrays.update({f"{name}_indices": values for name, values in split.items()})
    return result, arrays


def run_study(config, output):
    validate_config(config)
    # Bound BLAS parallelism for predictable small-network timing and results.
    with threadpool_limits(limits=1):
        output, metadata = create_run(output, config)
        start = perf_counter()
        (output / "cases").mkdir()
        results = []
        for level_index, level in enumerate(config["mismatch_levels"]):
            for seed in config["seeds"]:
                case_id = f"level_{level_index:02d}_seed_{seed}"
                case, arrays = run_case(config, level, seed)
                case["case_id"] = case_id
                write_json(output / "cases" / f"{case_id}.json", case)
                np.savez_compressed(output / "cases" / f"{case_id}.npz", **arrays)
                results.append(case)
                print(f"{case_id}: validation selected {case['selected_method']}", flush=True)
        write_json(output / "results.json", results)
        from adaptive_ode.experiments.reporting import write_report
        write_report(output, config, results)
        failures = sum(row["status"] != "ok" for case in results for row in case["tuning_attempts"])
        failures += sum(row["status"] != "ok" for case in results for row in case["methods"]
                        if not row["eligible_for_selection"])
        finish_run(output, metadata, perf_counter() - start, cases=len(results), failed_attempts=failures)
        print(f"Saved {len(results)} cases to {output}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    run_study(config, arguments.output)


if __name__ == "__main__":
    main()
