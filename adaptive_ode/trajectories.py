"""Train/validation/test experiments with initial-state transfer and frozen selection."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np

from adaptive_ode.assessment import training_scales
from adaptive_ode.core import BenchmarkConfig
from adaptive_ode.methods import classical_suite, initial_state_candidate, learned_polynomial_candidate
from adaptive_ode.numerics import time_grid, trajectory
from adaptive_ode.selection import choose, describe


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    states: np.ndarray
    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Each trajectory requires a nonempty experiment name.")
        times = time_grid(self.t).copy()
        states = trajectory(self.states, len(times)).copy()
        times.setflags(write=False)
        states.setflags(write=False)
        object.__setattr__(self, "t", times)
        object.__setattr__(self, "states", states)

    @property
    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(str(self.states.shape).encode())
        digest.update(self.t.tobytes())
        digest.update(self.states.tobytes())
        return digest.hexdigest()


def _metrics(actual, predicted, scales):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        residual = actual[1:] - predicted[1:]  # supplied initial state is not a forecast
        values = {"mse": float(np.mean(residual**2)),
                  "normalized_mse": float(np.mean((residual/scales)**2))}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite episode score.")
    return values


def _aggregate(episodes):
    mse = float(np.mean([item["mse"] for item in episodes]))
    normalized = float(np.sqrt(np.mean([item["normalized_mse"] for item in episodes])))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "normalized_rmse": normalized}


@dataclass
class TrajectoryBenchmarkResult:
    config: BenchmarkConfig
    datasets: dict
    scales: np.ndarray
    rows: list
    trials: list
    predictions: dict
    selected_method: str | None
    models: dict = field(repr=False)
    context: dict = field(default_factory=dict)

    @property
    def ranking(self):
        return sorted((dict(row) for row in self.rows), key=lambda row: (
            row["validation_score"] is None,
            row["validation_score"] if row["validation_score"] is not None else float("inf")))

    @property
    def assessment(self):
        return {
            "protocol": "independent trajectories; fixed initial-state transfer; no test refitting",
            "selection_rule": describe(self.config),
            "selected_method": self.selected_method,
            "no_recommendation_reason": (None if self.selected_method else
                "No successful validation candidate meets the policy; no fallback winner was chosen."),
            "trajectory_counts": {role: len(records) for role, records in self.datasets.items()},
            "normalization_scales": self.scales.tolist(),
            "weighting": "equal weight per trajectory, irrespective of record length",
            "timing": "median repeated validation-batch prediction time; fitting measured once; search cost includes all repeats and variants",
            "limitations": [
                "Each record must be an independent experiment, not an overlapping window from another role.",
                "Known-equation integration and learned dynamics are different tasks; observation error does not isolate numerical error.",
                "Only the first measured state initializes validation/test predictions; its measurement error can affect the entire rollout.",
                "No validation/test parameter updates or state resets occur within a trajectory.",
                "Test failures never trigger reselection. Partial-success averages are not reported as full performance.",
                "Bootstrap intervals resample whole test trajectories with the fitted model and selection fixed; they require independent experiments and do not include training/calibration uncertainty.",
                "Intervals from a few trajectories or identical observed scores can be narrow or degenerate without establishing certainty on future experiments.",
                "Targets may be noisy; no stiffness, chaos or latent-state sufficiency is inferred.",
                "Public research validation on independent real datasets remains necessary.",
            ],
        }

    def report(self):
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        def number(value):
            return "unavailable" if value is None else f"{value:.6g}"
        lines = ["# Independent-trajectory method comparison", "",
                 f"Selected method: **{cell(self.selected_method or 'none')}**.",
                 f"Rule: {describe(self.config)}.", "",
                 "Training, validation and test contain separate experiments. Selection is frozen before any test prediction.",
                 f"Trajectory counts: {self.assessment['trajectory_counts']}.",
                 f"Training-derived scales: {self.scales.tolist()}.", "",
                 "The first observation initializes each rollout and is excluded from error scores. "
                 "Metrics weight experiments equally. Prediction cost covers the validation batch; fitting is reported separately.", "",
                 "| Method | Chosen variant | Meets target | Validation score | Test score | Test status | Median prediction seconds | Fit seconds |",
                 "|---|---|---|---:|---:|---|---:|---:|"]
        for row in self.ranking:
            lines.append(f"| {cell(row['method'])} | {cell(row['variant'])} | {row['meets_accuracy_target']} | "
                         f"{number(row['validation_score'])} | {number(row['test_score'])} | {row['test_status']} | "
                         f"{number(row['prediction_seconds'])} | {number(row['fit_seconds'])} |")
        if not self.selected_method:
            lines.extend(["", self.assessment["no_recommendation_reason"]])
        lines.extend(["", "## Test uncertainty", ""])
        for row in self.ranking:
            lines.append(f"- {cell(row['method'])}: whole-trajectory 95% bootstrap interval for test "
                         f"{self.config.selection_metric}: {row['test_score_interval'] or 'unavailable'}.")
        lines.extend(["", "## Scope and limitations", ""])
        lines.extend(f"- {item}" for item in self.assessment["limitations"])
        lines.extend(["", "## Diagnostics", ""])
        for trial in self.trials:
            if trial["message"] or trial["warnings"]:
                lines.append(f"- {cell(trial['variant'])}: {cell(trial['message'])} {cell('; '.join(trial['warnings']))}")
        for row in self.rows:
            for episode in row["test_episodes"]:
                if episode.get("message"):
                    lines.append(f"- {cell(row['method'])}, test {cell(episode['name'])}: {cell(episode['message'])}")
        return "\n".join(lines) + "\n"

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        arrays, dataset_manifest = dict(self.predictions), {}
        for role, records in self.datasets.items():
            dataset_manifest[role] = []
            for index, record in enumerate(records):
                prefix = f"{role}_{index:03d}"
                arrays[f"{prefix}_t"] = record.t
                arrays[f"{prefix}_states"] = record.states
                dataset_manifest[role].append({"name": record.name, "array_prefix": prefix,
                                               "samples": len(record.t), "sha256": record.fingerprint})
        import importlib.metadata
        versions = {}
        for package in ("adaptive-solver-bench", "numpy", "scipy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        record = {"schema_version": 3, "evaluation_kind": "independent_trajectories",
                  "config": asdict(self.config), "datasets": dataset_manifest,
                  "normalization_scales": self.scales.tolist(), "selected_method": self.selected_method,
                  "rows": self.rows, "trials": self.trials, "assessment": self.assessment, "context": self.context,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "environment": {"python": platform.python_version(), "platform": platform.platform(), "versions": versions}}
        (path / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        columns = ["method", "family", "variant", "selected", "status", "meets_accuracy_target", "validation_score",
                   "test_score", "test_status", "fit_seconds", "prediction_seconds", "fit_plus_prediction_seconds", "search_seconds"]
        with (path / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.ranking)
        (path / "report.md").write_text(self.report())
        np.savez_compressed(path / "trajectories.npz", **arrays)
        return path


def benchmark_trajectories(*, train, validation, test, config=None, candidates=None, rhs=None):
    """Fit on full training experiments, select on validation experiments, report test.

    Train/validation fractions do not apply here: roles are supplied explicitly.
    Candidate.fit_trajectories receives lists of training time/state arrays only.
    Validation/test callbacks receive times and one initial state, never targets.
    """
    config = config or BenchmarkConfig(selection_metric="normalized_rmse")
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    datasets = {"train": list(train), "validation": list(validation), "test": list(test)}
    if any(not records for records in datasets.values()):
        raise ValueError("Supply at least one independent trajectory in each role.")
    all_records = [record for records in datasets.values() for record in records]
    if any(not isinstance(record, Trajectory) for record in all_records):
        raise TypeError("All records must be Trajectory instances.")
    if len({record.name for record in all_records}) != len(all_records):
        raise ValueError("Experiment names must be unique across all roles.")
    if len({record.fingerprint for record in all_records}) != len(all_records):
        raise ValueError("Duplicate trajectory contents detected; do not reuse experiments across roles.")
    dimension = all_records[0].states.shape[1]
    if any(record.states.shape[1] != dimension for record in all_records):
        raise ValueError("All trajectories must have the same state dimension and column meaning.")
    fitted_data = [(record.t[::config.train_stride], record.states[::config.train_stride]) for record in datasets["train"]]
    if any(len(times) < 2 for times, _ in fitted_data):
        raise ValueError("Each training trajectory needs at least two retained samples.")
    scales = training_scales(np.concatenate([states for _, states in fitted_data]))
    if rhs is not None and (not callable(rhs) or candidates is not None):
        raise ValueError("Provide a callable rhs for default candidates, or explicit candidates, not both.")
    if candidates is None:
        if config.include_ml:
            raise ValueError("Default neural time-to-state models cannot transfer to new initial states. Supply a transferable custom candidate instead.")
        candidates = [initial_state_candidate(), learned_polynomial_candidate(degree=1, max_nfev=config.max_nfev),
                      learned_polynomial_candidate(max_nfev=config.max_nfev)]
        if rhs is not None:
            base_step = max(float(np.median(np.diff(times))) for times, _ in fitted_data)
            candidates = classical_suite(rhs, max_nfev=config.max_nfev, sweep=config.solver_sweep, base_step=base_step) + candidates
    candidates = list(candidates)
    if not candidates or any(not isinstance(item.name, str) or not item.name for item in candidates):
        raise ValueError("Supply candidates with nonempty names.")
    if len({item.name for item in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique.")
    if any(item.group is not None and (not isinstance(item.group, str) or not item.group) for item in candidates):
        raise ValueError("Candidate groups must be nonempty strings.")
    trials, predictions, models = [], {}, {}
    for index, candidate in enumerate(candidates):
        row = {"method": candidate.group or candidate.name, "variant": candidate.name, "family": candidate.family,
               "status": "failed", "message": "", "warnings": [], "details": {}, "fit_seconds": 0.0,
               "prediction_seconds": 0.0, "prediction_timings": [], "validation_score": None,
               "validation_mse": None, "validation_normalized_rmse": None, "validation_episodes": [],
               "prediction_key": f"prediction_{index:03d}"}
        started, fitted = perf_counter(), False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if candidate.fit_trajectories is None:
                    raise ValueError("Candidate has no fit_trajectories callback; independent-trajectory fitting is unsupported.")
                model = candidate.fit_trajectories([t.copy() for t, _ in fitted_data], [y.copy() for _, y in fitted_data])
                row["fit_seconds"] = perf_counter() - started
                fitted = True
                if model.transfer is None:
                    raise ValueError("Fitted model cannot transfer to a new initial state.")
                outputs = []
                for repeat in range(config.timing_repeats):
                    started_prediction = perf_counter()
                    current = [model.predict_from_initial(record.t.copy(), record.states[0].copy()) for record in datasets["validation"]]
                    row["prediction_timings"].append(perf_counter() - started_prediction)
                    if repeat == 0:
                        outputs = [array.copy() for array in current]
                    elif any(not np.allclose(a, b, rtol=1e-12, atol=1e-12) for a, b in zip(outputs, current)):
                        raise ValueError("Prediction changed across timing repeats; model must not mutate fitted state.")
                row["prediction_seconds"] = float(np.median(row["prediction_timings"]))
                for episode_index, (record, predicted) in enumerate(zip(datasets["validation"], outputs)):
                    if predicted.shape != record.states.shape:
                        raise ValueError("Prediction shape differs from observed states.")
                    metrics = _metrics(record.states, predicted, scales)
                    row["validation_episodes"].append({"name": record.name, **metrics})
                    predictions[f"{row['prediction_key']}_validation_{episode_index:03d}"] = predicted
                metrics = _aggregate(row["validation_episodes"])
                row.update(validation_mse=metrics["mse"], validation_normalized_rmse=metrics["normalized_rmse"],
                           validation_score=metrics[config.selection_metric], status="ok", details=model.details)
                models[candidate.name] = model
            except Exception as error:
                if not fitted:
                    row["fit_seconds"] = perf_counter() - started
                row["message"] = f"{type(error).__name__}: {error}"
            row["warnings"] = [f"{item.category.__name__}: {item.message}" for item in caught]
        row["fit_plus_prediction_seconds"] = row["fit_seconds"] + row["prediction_seconds"]
        row["search_seconds"] = row["fit_seconds"] + sum(row["prediction_timings"])
        row["meets_accuracy_target"] = row["status"] == "ok" and (config.accuracy_target is None or row["validation_score"] <= config.accuracy_target)
        trials.append(row)
    rows = []
    for group in dict.fromkeys(row["method"] for row in trials):
        variants = [row for row in trials if row["method"] == group]
        best = choose(variants, config, representative=True) or variants[0]
        row = dict(best)
        row["search_seconds"] = sum(item["search_seconds"] for item in variants)
        rows.append(row)
    chosen = choose(rows, config)
    winner = chosen["method"] if chosen else None  # frozen before any test prediction
    selected_models = {}
    for row in rows:
        row.update(selected=row["method"] == winner, test_status="not_evaluated", test_score=None,
                   test_mse=None, test_rmse=None, test_normalized_rmse=None, test_episodes=[], test_score_interval=None)
        if row["status"] != "ok":
            continue
        model = models[row["variant"]]
        selected_models[row["method"]] = model
        for episode_index, record in enumerate(datasets["test"]):
            try:
                predicted = model.predict_from_initial(record.t.copy(), record.states[0].copy())
                if predicted.shape != record.states.shape:
                    raise ValueError("Prediction shape differs from observed states.")
                metrics = _metrics(record.states, predicted, scales)
                predictions[f"{row['prediction_key']}_test_{episode_index:03d}"] = predicted
                row["test_episodes"].append({"name": record.name, "status": "ok", **metrics})
            except Exception as error:
                row["test_episodes"].append({"name": record.name, "status": "failed", "message": f"{type(error).__name__}: {error}"})
        if any(item["status"] != "ok" for item in row["test_episodes"]):
            row["test_status"] = "failed"
            continue
        metrics = _aggregate(row["test_episodes"])
        row.update(test_status="ok", test_score=metrics[config.selection_metric], test_mse=metrics["mse"],
                   test_rmse=metrics["rmse"], test_normalized_rmse=metrics["normalized_rmse"])
        if len(row["test_episodes"]) >= 2:
            values = np.array([episode["mse" if config.selection_metric == "mse" else "normalized_mse"] for episode in row["test_episodes"]])
            rng = np.random.default_rng(config.seed)
            boot = np.array([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(1000)])
            if config.selection_metric == "normalized_rmse":
                boot = np.sqrt(boot)
            row["test_score_interval"] = np.quantile(boot, [.025, .975]).tolist()
    return TrajectoryBenchmarkResult(config, datasets, scales, rows, trials, predictions, winner, selected_models,
                                     {"input_kind": "independent_trajectories", "equations_supplied": rhs is not None,
                                      "selection_frozen_before_test": True})
