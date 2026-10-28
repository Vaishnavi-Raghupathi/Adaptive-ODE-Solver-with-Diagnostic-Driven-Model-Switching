"""Validation-only method selection and serializable evaluation results."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Callable
import warnings
from numbers import Real

import numpy as np

from adaptive_ode.evaluation.metrics import compute_metrics
from adaptive_ode.numerics import time_grid, trajectory
from adaptive_ode.selection import choose


@dataclass(frozen=True)
class BenchmarkConfig:
    """Chronological fractions, observation count, and fixed ML fitting budget."""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    num_points: int = 301
    seed: int = 7
    include_ml: bool = False
    nn_max_iter: int = 1200
    max_nfev: int = 50000
    train_stride: int = 1
    selection_metric: str = "mse"
    selection_policy: str = "min_error"
    accuracy_target: float | None = None
    selection_cost: str = "prediction_seconds"
    timing_repeats: int = 1
    solver_sweep: bool = False

    def __post_init__(self):
        fractions = np.array([self.train_fraction, self.validation_fraction])
        if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0) or fractions.sum() >= 1:
            raise ValueError("Training and validation fractions must be positive and sum to less than 1.")
        for name, minimum in [("num_points", 12), ("nn_max_iter", 1), ("max_nfev", 4), ("train_stride", 1)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer.")
        if not isinstance(self.include_ml, bool):
            raise ValueError("include_ml must be a boolean.")
        if self.selection_metric not in ("mse", "normalized_rmse"):
            raise ValueError("selection_metric must be 'mse' or 'normalized_rmse'.")
        if self.selection_policy not in ("min_error", "fastest_within_target"):
            raise ValueError("selection_policy must be 'min_error' or 'fastest_within_target'.")
        if self.selection_cost not in ("prediction_seconds", "fit_plus_prediction_seconds"):
            raise ValueError("selection_cost must be 'prediction_seconds' or 'fit_plus_prediction_seconds'.")
        if self.accuracy_target is not None and (isinstance(self.accuracy_target, bool)
                or not isinstance(self.accuracy_target, Real) or not np.isfinite(self.accuracy_target)
                or self.accuracy_target < 0):
            raise ValueError("accuracy_target must be finite and nonnegative.")
        if self.selection_policy == "fastest_within_target" and self.accuracy_target is None:
            raise ValueError("fastest_within_target requires an accuracy_target.")
        if self.selection_policy == "min_error" and self.accuracy_target is not None:
            raise ValueError("accuracy_target requires selection_policy='fastest_within_target'.")
        if self.accuracy_target is not None:
            object.__setattr__(self, "accuracy_target", float(self.accuracy_target))
        if isinstance(self.timing_repeats, bool) or not isinstance(self.timing_repeats, int) or not 1 <= self.timing_repeats <= 20:
            raise ValueError("timing_repeats must be an integer from 1 to 20.")
        if not isinstance(self.solver_sweep, bool):
            raise ValueError("solver_sweep must be a boolean.")


def chronological_indices(n, config):
    a = int(n * config.train_fraction)
    b = a + int(n * config.validation_fraction)
    if min(a, b-a, n-b) < 2:
        raise ValueError("Each chronological split must contain at least two samples.")
    return {"train": np.arange(a), "validation": np.arange(a, b), "test": np.arange(b, n)}


@dataclass
class FittedModel:
    """Predict from the original initial state; optionally transfer to a new one.

    A prediction callback receives times only. A transfer callback receives a
    new full time grid and initial state, never observations from that trajectory.
    """

    predict: Callable
    details: dict = field(default_factory=dict)
    transfer: Callable | None = None

    def predict_from_initial(self, t, y0):
        if self.transfer is None:
            raise NotImplementedError("This trajectory model does not support new initial conditions.")
        t = time_grid(t)
        return trajectory(self.transfer(t, np.asarray(y0, dtype=float)), len(t))


@dataclass(frozen=True)
class Candidate:
    """An independently named trial; trials sharing a group are tuned on validation.

    fit receives ONLY training times/states and returns a FittedModel. Custom
    callbacks must not capture held-out labels from the caller's environment.
    """

    name: str
    family: str
    fit: Callable
    group: str | None = None
    fit_trajectories: Callable | None = None


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    name: str
    t: np.ndarray
    observed: np.ndarray
    target: np.ndarray
    split: dict
    rows: list
    trials: list
    predictions: dict
    selected_method: str | None
    models: dict = field(repr=False)
    reference_kind: str = "observations"
    context: dict = field(default_factory=dict)

    @property
    def ranking(self):
        """Copies ordered by the configured validation score; test errors never rank."""
        return sorted((dict(row) for row in self.rows), key=lambda row: (
            row["validation_score"] is None,
            row["validation_score"] if row["validation_score"] is not None else float("inf"),
        ))

    @property
    def assessment(self):
        """Training-data profile and evidence for this benchmark's recommendation."""
        from adaptive_ode.assessment import assess_result
        return assess_result(self)

    def report(self):
        """A portable Markdown recommendation with data and evaluation limitations."""
        from adaptive_ode.assessment import render_report
        return render_report(self)

    def save(self, path):
        """Export JSON, CSV and safe numeric NPZ; fitted models are not pickled."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        versions = {}
        for name in ["adaptive-solver-bench", "numpy", "scipy", "scikit-learn"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        record = {
            "schema_version": 2, "name": self.name, "config": asdict(self.config),
            "selected_method": self.selected_method, "reference_kind": self.reference_kind,
            "context": self.context,
            "rows": self.rows, "trials": self.trials,
            "split_indices": {name: values.tolist() for name, values in self.split.items()},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "versions": versions},
            "assessment": self.assessment,
        }
        (path / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        columns = ["method", "family", "variant", "status", "test_status", "selected", "meets_accuracy_target", "validation_score", "validation_mse",
                   "validation_normalized_rmse", "test_normalized_rmse", "test_mse", "test_rmse",
                   "test_mae", "fit_seconds", "prediction_seconds", "fit_plus_prediction_seconds", "search_seconds", "message"]
        with (path / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.ranking)
        arrays = {"t": self.t, "observed": self.observed, "target": self.target, **self.predictions}
        arrays.update({f"split_{name}": indices for name, indices in self.split.items()})
        np.savez_compressed(path / "predictions.npz", **arrays)
        (path / "report.md").write_text(self.report())
        return path

    def plot(self, state=0):
        """Return a Matplotlib figure; install the plots extra to use this method."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("Plotting requires: pip install 'adaptive-solver-bench[plots]'") from error
        if not isinstance(state, int) or not 0 <= state < self.target.shape[1]:
            raise ValueError("state must be a valid state-dimension index.")
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.plot(self.t, self.target[:, state], color="black", label=self.reference_kind)
        for row in self.ranking:
            if row["status"] == "ok":
                ax.plot(self.t, self.predictions[row["prediction_key"]][:, state], label=row["method"], alpha=0.7)
        ax.axvline(self.t[self.split["validation"][0]], color="gray", linestyle=":", label="validation starts")
        ax.axvspan(self.t[self.split["test"][0]], self.t[-1], color="gray", alpha=0.12, label="final test")
        ax.set(xlabel="Time", ylabel=f"State {state}", title=self.name)
        ax.legend(fontsize=8)
        return fig


def evaluate(t, observed, candidates, *, config=None, reference=None, name="Trajectory benchmark"):
    """Fit on training, tune/select on validation, then score untouched test rows.

    A clean reference, when supplied, is used only for final reporting; validation
    always uses observations. Predictions integrate from the original initial
    state. A single thread is recommended for timing comparisons involving ML.
    """
    config = config or BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    t = time_grid(t).copy()
    observed = trajectory(observed, len(t)).copy()
    target = observed.copy() if reference is None else trajectory(reference, len(t)).copy()
    if target.shape != observed.shape:
        raise ValueError("Reference and observation state dimensions must agree.")
    candidates = list(candidates)
    names = [candidate.name for candidate in candidates]
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Provide at least one candidate with a unique nonempty name.")
    if any(candidate.group is not None and (not isinstance(candidate.group, str) or not candidate.group)
           for candidate in candidates):
        raise ValueError("Candidate groups must be nonempty strings.")
    split = chronological_indices(len(t), config)
    split["fit"] = split["train"][::config.train_stride]
    if len(split["fit"]) < 2:
        raise ValueError("At least two retained training samples are required.")
    from adaptive_ode.assessment import training_scales
    scales = training_scales(observed[split["fit"]])

    def normalized_rmse(actual, predicted):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value = float(np.sqrt(np.mean(((actual - predicted) / scales) ** 2)))
        if not np.isfinite(value):
            raise ValueError("Normalized error is non-finite.")
        return value

    trials, predictions, models = [], {}, {}
    for index, candidate in enumerate(candidates):
        row = {
            "method": candidate.group or candidate.name, "variant": candidate.name,
            "family": candidate.family, "status": "failed", "message": "",
            "validation_mse": None, "validation_normalized_rmse": None, "validation_score": None,
            "fit_seconds": 0.0, "prediction_seconds": 0.0, "prediction_timings": [],
            "prediction_key": f"prediction_{index:03d}", "details": {}, "warnings": [],
        }
        fit_start = perf_counter()
        fitted = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = candidate.fit(t[split["fit"]].copy(), observed[split["fit"]].copy())
                row["fit_seconds"] = perf_counter() - fit_start
                fitted = True
                for repeat in range(config.timing_repeats):
                    predict_start = perf_counter()
                    current = trajectory(model.predict(t.copy()), len(t))
                    row["prediction_timings"].append(perf_counter() - predict_start)
                    if repeat == 0:
                        predicted = current.copy()
                    elif not np.allclose(current, predicted, rtol=1e-12, atol=1e-12):
                        raise ValueError("Prediction changed across timing repeats; predictors must be deterministic and not mutate fitted state.")
                row["prediction_seconds"] = float(np.median(row["prediction_timings"]))
                if predicted.shape != observed.shape:
                    raise ValueError("Candidate prediction shape differs from observations.")
                row["details"] = model.details
                row["validation_mse"] = compute_metrics(observed[split["validation"]], predicted[split["validation"]])["mse"]
                if not np.isfinite(row["validation_mse"]):
                    raise ValueError("Validation error is non-finite.")
                row["validation_normalized_rmse"] = normalized_rmse(
                    observed[split["validation"]], predicted[split["validation"]])
                row["validation_score"] = row[f"validation_{config.selection_metric}"]
                row["status"] = "ok"
                predictions[row["prediction_key"]] = predicted.copy()
                models[candidate.name] = model
            except Exception as error:
                if fitted:
                    row["prediction_seconds"] = perf_counter() - predict_start
                else:
                    row["fit_seconds"] = perf_counter() - fit_start
                row["validation_mse"] = None
                row["validation_normalized_rmse"] = None
                row["validation_score"] = None
                row["message"] = f"{type(error).__name__}: {error}"
            row["warnings"] = [f"{item.category.__name__}: {item.message}" for item in caught]
        row["fit_plus_prediction_seconds"] = row["fit_seconds"] + row["prediction_seconds"]
        row["search_seconds"] = row["fit_seconds"] + (sum(row["prediction_timings"]) or row["prediction_seconds"])
        row["meets_accuracy_target"] = (row["status"] == "ok" and
                (config.accuracy_target is None or row["validation_score"] <= config.accuracy_target))
        trials.append(row)

    rows = []
    for group in dict.fromkeys(row["method"] for row in trials):
        variants = [row for row in trials if row["method"] == group]
        successful = [row for row in variants if row["status"] == "ok"]
        best = choose(successful, config, representative=True) if successful else variants[0]
        row = dict(best)
        row["search_seconds"] = sum(item["search_seconds"] for item in variants)
        rows.append(row)
    successful = [row for row in rows if row["status"] == "ok"]
    chosen = choose(successful, config)
    winner = chosen["method"] if chosen else None
    # Only this reporting stage touches final test targets.
    selected_models = {}
    for row in rows:
        row.update(selected=row["method"] == winner, test_status="not_evaluated", test_mse=None,
                   test_rmse=None, test_mae=None, test_normalized_rmse=None, residual_summary=None)
        if row["status"] == "ok":
            prediction = predictions[row["prediction_key"]][split["test"]]
            try:
                with np.errstate(over="raise", invalid="raise"):
                    metrics = compute_metrics(target[split["test"]], prediction)
                    normalized = normalized_rmse(target[split["test"]], prediction)
                    residual = target[split["test"]] - prediction
                    residual_summary = {"mean_by_state": residual.mean(axis=0).tolist(),
                                        "max_abs_by_state": np.abs(residual).max(axis=0).tolist()}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise ValueError("Non-finite test metric.")
                row.update({f"test_{key}": value for key, value in metrics.items()})
                row.update(test_status="ok", residual_summary=residual_summary, test_normalized_rmse=normalized)
            except (FloatingPointError, ValueError) as error:
                # Reporting failure must never select a different winner.
                row.update(test_status="failed", message=f"Final test scoring failed: {error}")
            selected_models[row["method"]] = models[row["variant"]]
    return BenchmarkResult(config, name, t, observed, target, split, rows, trials, predictions,
                           winner, selected_models, "supplied reference" if reference is not None else "observations")
