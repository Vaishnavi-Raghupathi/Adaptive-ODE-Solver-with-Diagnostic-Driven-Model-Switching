"""Plotting utilities for visualization."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(t, y_true, y_pred, labels=None, figsize=(12, 8)):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    n_dims = y_true.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    if n_dims == 1:
        axes = [axes]
    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, y_true[:, i], 'b-', linewidth=2, label='True')
        ax.plot(t, y_pred[:, i], 'r--', linewidth=2, label='Predicted')
        ax.set_ylabel(labels[i] if labels else f"y{i}", fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle("ODE Trajectory: True vs Predicted", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, axes


def plot_residuals(t, residuals, labels=None, figsize=(12, 6)):
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
    n_dims = residuals.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    if n_dims == 1:
        axes = [axes]
    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, residuals[:, i], 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"Residual ({labels[i]})" if labels else f"Residual (y{i})", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle("Prediction Residuals", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, axes


def plot_solution(t, y, labels=None, title="ODE Solution", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    if y.ndim == 1:
        ax.plot(t, y)
    else:
        for i in range(y.shape[0]):
            ax.plot(t, y[i], label=labels[i] if labels else f"y{i}")
        ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def compare_models(y_true, y_classical, y_neural):
    classical_mse = np.mean((y_true - y_classical) ** 2)
    neural_mse    = np.mean((y_true - y_neural) ** 2)
    improvement   = ((classical_mse - neural_mse) / classical_mse * 100.0
                     if classical_mse > 0 else 0.0)
    print(f"Classical MSE: {classical_mse:.6f}")
    print(f"Neural ODE MSE: {neural_mse:.6f}")
    print(f"Improvement (%): {improvement:.2f}")
    return {
        "classical_mse":    classical_mse,
        "neural_mse":       neural_mse,
        "classical_rmse":   np.sqrt(classical_mse),
        "neural_rmse":      np.sqrt(neural_mse),
        "improvement_percent": improvement,
    }


def plot_model_comparison(t, y_true, y_classical, y_neural,
                          labels=None, figsize=(12, 9)):
    """
    Plot true, classical, and neural trajectories.
    Works for any number of dimensions (2D oscillator or 3D Lorenz).
    """
    y_true     = np.asarray(y_true)
    y_classical = np.asarray(y_classical)
    y_neural   = np.asarray(y_neural)

    if y_true.ndim != 2:
        raise ValueError("y_true must be 2D array (n_points, n_dims)")
    if y_true.shape != y_classical.shape or y_true.shape != y_neural.shape:
        raise ValueError("y_true, y_classical, y_neural must have the same shape")

    n_dims = y_true.shape[1]

    # Default labels: x/y/z for 3D, position/velocity for 2D, else dim_0/dim_1...
    if labels is None:
        if n_dims == 3:
            labels = ["x", "y", "z"]
        elif n_dims == 2:
            labels = ["position", "velocity"]
        else:
            labels = [f"dim_{i}" for i in range(n_dims)]

    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, y_true[:, i],      'k-',  label='True')
        ax.plot(t, y_classical[:, i], 'r--', label='Classical')
        ax.plot(t, y_neural[:, i],    'b-',  label='Neural ODE')
        ax.set_title(f"{labels[i]}-dimension")
        ax.set_ylabel("State")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig, axes