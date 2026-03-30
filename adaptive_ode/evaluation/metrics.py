"""Evaluation metrics for model predictions."""

import numpy as np


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics between true and predicted trajectories.

    Metrics are computed across all samples and dimensions.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth values (shape: n_samples, n_dims or n_samples).
    y_pred : ndarray
        Predicted values (same shape as y_true).

    Returns
    -------
    dict
        Dictionary with metrics:
        {
            "mse": float,
            "rmse": float,
            "mae": float
        }

    Raises
    ------
    ValueError
        If y_true and y_pred do not have the same shape.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def compare_models(y_true, y_classical, y_neural):
    """
    Compare classical and Neural ODE predictions using regression metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth values (shape: n_samples, n_dims or n_samples).
    y_classical : ndarray
        Classical model predictions (same shape as y_true).
    y_neural : ndarray
        Neural ODE predictions (same shape as y_true).

    Returns
    -------
    tuple
        (classical_metrics, neural_metrics)
        where each is a dict with keys: "mse", "rmse", "mae".
    """
    classical_metrics = compute_metrics(y_true, y_classical)
    neural_metrics = compute_metrics(y_true, y_neural)

    if classical_metrics["mse"] > 0:
        improvement_percent = (
            (classical_metrics["mse"] - neural_metrics["mse"])
            / classical_metrics["mse"]
        ) * 100.0
    else:
        improvement_percent = 0.0

    print("Classical Model:")
    print(f"  MSE: {classical_metrics['mse']:.6f}")
    print(f"  RMSE: {classical_metrics['rmse']:.6f}")
    print(f"  MAE: {classical_metrics['mae']:.6f}")

    print("\nNeural ODE:")
    print(f"  MSE: {neural_metrics['mse']:.6f}")
    print(f"  RMSE: {neural_metrics['rmse']:.6f}")
    print(f"  MAE: {neural_metrics['mae']:.6f}")

    print("\nImprovement:")
    print(f"  % improvement in MSE: {improvement_percent:.2f}%")

    return classical_metrics, neural_metrics
