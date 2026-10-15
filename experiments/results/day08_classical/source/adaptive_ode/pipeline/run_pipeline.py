"""Adaptive solver-selection pipeline."""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_ode.diagnostics.engine import DiagnosticEngine
from adaptive_ode.solvers.candidates import (
    choose_best,
    run_cubic_spline_data,
    run_hybrid_residual,
    run_mlp_trajectory_data,
    run_pinn_surrogate,
    run_rk4,
    run_scipy_method,
    run_sindy_polynomial_data,
    solve_reference,
)
from adaptive_ode.systems import get_system


CLASSICAL_METHODS = ["RK45", "DOP853", "BDF", "Radau", "LSODA"]


def _fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _mask_train_test(n_points, train_fraction, seed):
    n_train = int(np.clip(train_fraction, 0.2, 0.9) * n_points)
    n_train = min(max(n_train, 2), n_points - 1)
    train_mask = np.zeros(n_points, dtype=bool)
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(n_points, size=n_train, replace=False)
    train_mask[train_idx] = True
    train_mask[0] = True
    return train_mask, ~train_mask


def _chronological_train_test(n_points, train_fraction):
    split_idx = int(np.clip(train_fraction, 0.2, 0.9) * n_points)
    split_idx = min(max(split_idx, 2), n_points - 1)
    train_mask = np.zeros(n_points, dtype=bool)
    train_mask[:split_idx] = True
    return train_mask, ~train_mask


def _result_row(result, recommended_name):
    metrics = result.metrics if result.metrics and np.isfinite(result.metrics["mse"]) else None
    row = {
        "Solver": result.name,
        "Family": result.family,
        "Status": result.status,
        "MSE": None,
        "RMSE": None,
        "MAE": None,
        "Runtime (s)": round(result.runtime_seconds, 4),
        "Recommended": result.name == recommended_name,
        "Notes": result.message,
    }
    if metrics:
        row["MSE"] = metrics["mse"]
        row["RMSE"] = metrics["rmse"]
        row["MAE"] = metrics["mae"]
    return row


def _recommendation_reason(best, system):
    if best is None:
        return "No solver completed successfully."

    reason = (
        f"{best.name} produced the strongest held-out accuracy for this "
        f"{system.stiffness} system."
    )
    if best.family == "Hybrid correction":
        return reason + " The residual learner helped because the baseline physics has systematic error."
    if best.family == "Physics-informed NN":
        return reason + " The physics penalty helped the neural surrogate respect the governing equations."
    if best.family == "Classical implicit":
        return reason + " Implicit integration is usually a strong fit when stiffness is present."
    return reason + " The explicit numerical method is accurate enough without extra model training."


def _data_recommendation_reason(best):
    if best is None:
        return "No data-driven candidate completed successfully."

    reason = f"{best.name} produced the lowest held-out error on the uploaded trajectory."
    if best.family == "Data-driven dynamics":
        return reason + " It also provides an explicit learned ODE that can be integrated forward."
    if best.family == "Data-driven NN":
        return reason + " This is useful for trajectory fitting when the governing equation is unknown."
    return reason + " This is strongest when the goal is smooth interpolation inside the observed time range."


def _plot_best_trajectory(t, y_reference, y_observed, best, labels):
    n_dims = y_reference.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(11, 2.6 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(t, y_reference[:, dim], color="black", linewidth=2, label="reference")
        ax.scatter(t, y_observed[:, dim], color="#7c9eb2", s=9, alpha=0.45, label="observed")
        if best and best.prediction is not None:
            ax.plot(t, best.prediction[:, dim], color="#c23b22", linestyle="--", linewidth=2, label=best.name)
        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time")
    fig.suptitle("Recommended Solver Trajectory", fontweight="bold")
    fig.tight_layout()
    return _fig_to_buf(fig)


def _plot_best_residuals(t, y_reference, best, labels):
    if best is None or best.prediction is None:
        return None

    residuals = y_reference - best.prediction
    n_dims = residuals.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(11, 2.4 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(t, residuals[:, dim], color="#2f7d32", linewidth=1.5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("time")
    fig.suptitle("Recommended Solver Residuals", fontweight="bold")
    fig.tight_layout()
    return _fig_to_buf(fig)


def _plot_candidate_mse(results):
    successful = [result for result in results if result.status == "ok" and result.metrics]
    if not successful:
        return None

    successful = sorted(successful, key=lambda result: result.metrics["mse"])
    names = [result.name for result in successful]
    values = [result.metrics["mse"] for result in successful]
    colors = ["#2f7d32" if idx == 0 else "#7c9eb2" for idx in range(len(successful))]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(names, values, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("held-out MSE (log scale)")
    ax.set_title("Solver Ranking", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return _fig_to_buf(fig)


def _candidate_predictions(results):
    return [
        {
            "name": result.name,
            "family": result.family,
            "status": result.status,
            "prediction": result.prediction,
        }
        for result in results
    ]


def _run_system_pipeline(system, config=None):
    config = config or {}

    noise_std = float(config.get("noise_std", 0.0))
    num_points = int(config.get("num_points", 300))
    train_fraction = float(config.get("train_fraction", 0.7))
    include_pinn = bool(config.get("include_pinn", True))
    pinn_epochs = int(config.get("pinn_epochs", config.get("epochs", 250)))
    save_plots = bool(config.get("save_plots", True))
    seed = int(config.get("seed", 7))

    t = np.linspace(system.t_span[0], system.t_span[1], num_points)
    y_reference = solve_reference(system, t)
    y_observed = y_reference.copy()
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        y_observed += rng.normal(0.0, noise_std, size=y_observed.shape)
        y_observed[0] = y_reference[0]

    train_mask, test_mask = _mask_train_test(num_points, train_fraction, seed)

    results = [run_rk4(system, t, y_reference, test_mask)]
    for method in CLASSICAL_METHODS:
        results.append(run_scipy_method(system, method, t, y_reference, test_mask))

    results.append(run_hybrid_residual(system, t, y_observed, y_reference, train_mask, test_mask))

    if include_pinn and system.key in {"lorenz", "vanderpol", "robertson", "mismatch"}:
        results.append(
            run_pinn_surrogate(
                system,
                t,
                y_observed,
                y_reference,
                train_mask,
                test_mask,
                epochs=pinn_epochs,
            )
        )

    best = choose_best(results)
    recommended_name = best.name if best else None
    rows = [_result_row(result, recommended_name) for result in results]
    rows = sorted(rows, key=lambda row: float("inf") if row["MSE"] is None else row["MSE"])

    diagnostics = None
    if best and best.prediction is not None:
        diag_engine = DiagnosticEngine(y_reference, best.prediction)
        diagnostics = {
            "test_results": diag_engine.run_all_tests(),
            "error_metrics": diag_engine.get_error_metrics(),
            "residual_stats": diag_engine.get_residual_statistics(),
        }

    plots = {
        "trajectory": _plot_best_trajectory(t, y_reference, y_observed, best, system.labels),
        "residuals": _plot_best_residuals(t, y_reference, best, system.labels),
        "model_comparison": _plot_candidate_mse(results),
    }

    if save_plots:
        os.makedirs("outputs", exist_ok=True)
        for plot_name, plot_buf in plots.items():
            if plot_buf is None:
                continue
            with open(os.path.join("outputs", f"{plot_name}.png"), "wb") as handle:
                handle.write(plot_buf.getvalue())

    return {
        "system": {
            "key": system.key,
            "name": system.name,
            "description": system.description,
            "stiffness": system.stiffness,
            "labels": system.labels,
            "t_span": system.t_span,
        },
        "decision": recommended_name,
        "recommended_solver": _result_row(best, recommended_name) if best else None,
        "recommendation_reason": _recommendation_reason(best, system),
        "candidate_results": rows,
        "candidate_predictions": _candidate_predictions(results),
        "diagnostics": diagnostics,
        "plots": plots,
        "t": t,
        "y_reference": y_reference,
        "y_observed": y_observed,
        "train_fraction": train_fraction,
        "noise_std": noise_std,
    }


def run_pipeline(config=None):
    """
    Run a solver-selection benchmark for a built-in dynamical system.

    Parameters
    ----------
    config : dict
        system_key      str    Built-in system key.
        noise_std       float  Observation noise used for training data.
        num_points      int    Number of trajectory points.
        train_fraction  float  Initial fraction used to train neural candidates.
        include_pinn    bool   Whether to run the PINN-style candidate.
        pinn_epochs     int    PINN training epochs.
        save_plots      bool   Save generated plots into outputs/.
    """
    config = config or {}
    legacy_mismatch = bool(config.get("mismatch", False))
    system_key = config.get("system_key") or ("mismatch" if legacy_mismatch else "lorenz")
    return _run_system_pipeline(get_system(system_key), config)


def run_custom_pipeline(system, config=None):
    """Run the standard known-equation benchmark for a custom system."""
    return _run_system_pipeline(system, config)


def run_data_pipeline(t, y_observed, labels=None, config=None):
    """
    Run solver selection when the researcher has trajectory data but no RHS.

    The uploaded data is treated as the evaluation target. Candidates are
    trained on an initial time segment and scored on held-out future points.
    """
    config = config or {}
    t = np.asarray(t, dtype=float)
    y_observed = np.asarray(y_observed, dtype=float)

    if y_observed.ndim == 1:
        y_observed = y_observed.reshape(-1, 1)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if len(t) != len(y_observed):
        raise ValueError("t and y_observed must have the same number of rows.")
    if len(t) < 12:
        raise ValueError("Upload at least 12 rows for train/test evaluation.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y_observed)):
        raise ValueError("Uploaded data contains non-finite values.")

    order = np.argsort(t)
    t = t[order]
    y_observed = y_observed[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time values must be unique and strictly increasing after sorting.")

    train_fraction = float(config.get("train_fraction", 0.7))
    save_plots = bool(config.get("save_plots", True))
    labels = labels or [f"state_{idx + 1}" for idx in range(y_observed.shape[1])]
    train_mask, test_mask = _chronological_train_test(len(t), train_fraction)

    results = [
        run_cubic_spline_data(t, y_observed, train_mask, test_mask),
        run_mlp_trajectory_data(t, y_observed, train_mask, test_mask),
        run_sindy_polynomial_data(t, y_observed, train_mask, test_mask),
    ]

    best = choose_best(results)
    recommended_name = best.name if best else None
    rows = [_result_row(result, recommended_name) for result in results]
    rows = sorted(rows, key=lambda row: float("inf") if row["MSE"] is None else row["MSE"])

    diagnostics = None
    if best and best.prediction is not None:
        diag_engine = DiagnosticEngine(y_observed, best.prediction)
        diagnostics = {
            "test_results": diag_engine.run_all_tests(),
            "error_metrics": diag_engine.get_error_metrics(),
            "residual_stats": diag_engine.get_residual_statistics(),
        }

    system_info = {
        "key": "uploaded",
        "name": "Uploaded trajectory data",
        "description": "Data-only mode: no governing equation was supplied.",
        "stiffness": "unknown",
        "labels": labels,
        "t_span": (float(t[0]), float(t[-1])),
    }

    plots = {
        "trajectory": _plot_best_trajectory(t, y_observed, y_observed, best, labels),
        "residuals": _plot_best_residuals(t, y_observed, best, labels),
        "model_comparison": _plot_candidate_mse(results),
    }

    if save_plots:
        os.makedirs("outputs", exist_ok=True)
        for plot_name, plot_buf in plots.items():
            if plot_buf is None:
                continue
            with open(os.path.join("outputs", f"{plot_name}.png"), "wb") as handle:
                handle.write(plot_buf.getvalue())

    return {
        "system": system_info,
        "decision": recommended_name,
        "recommended_solver": _result_row(best, recommended_name) if best else None,
        "recommendation_reason": _data_recommendation_reason(best),
        "candidate_results": rows,
        "candidate_predictions": _candidate_predictions(results),
        "diagnostics": diagnostics,
        "plots": plots,
        "t": t,
        "y_reference": y_observed,
        "y_observed": y_observed,
        "train_fraction": train_fraction,
        "noise_std": None,
    }


if __name__ == "__main__":
    output = run_pipeline({
        "system_key": "mismatch",
        "noise_std": 0.0,
        "save_plots": True,
        "include_pinn": False,
    })
    print(output["recommended_solver"])
