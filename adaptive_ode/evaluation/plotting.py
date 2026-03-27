"""Plotting utilities for visualization."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(t, y_true, y_pred, labels=None, figsize=(12, 8)):
    """
    Plot true and predicted trajectories for each dimension.
    
    Parameters
    ----------
    t : ndarray
        Time points (shape: n_points)
    y_true : ndarray
        True trajectory (shape: n_points, n_dims)
    y_pred : ndarray
        Predicted trajectory (shape: n_points, n_dims)
    labels : list, optional
        Labels for each dimension (e.g., ['x', 'y', 'z'])
    figsize : tuple
        Figure size (default: (12, 8))
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_dims = y_true.shape[1]
    
    # Create subplots
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    
    # Handle single dimension case
    if n_dims == 1:
        axes = [axes]
    
    # Plot each dimension
    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, y_true[:, i], 'b-', linewidth=2, label='True')
        ax.plot(t, y_pred[:, i], 'r--', linewidth=2, label='Predicted')
        
        # Set labels
        if labels:
            ax.set_ylabel(f"{labels[i]}", fontsize=11)
        else:
            ax.set_ylabel(f"y{i}", fontsize=11)
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle("ODE Trajectory: True vs Predicted", fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    return fig, axes


def plot_residuals(t, residuals, labels=None, figsize=(12, 6)):
    """
    Plot residuals for each dimension.
    
    Parameters
    ----------
    t : ndarray
        Time points (shape: n_points)
    residuals : ndarray
        Residuals (shape: n_points, n_dims)
    labels : list, optional
        Labels for each dimension (e.g., ['x', 'y', 'z'])
    figsize : tuple
        Figure size (default: (12, 6))
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Handle 1D case
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
    
    n_dims = residuals.shape[1]
    
    # Create subplots
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    
    # Handle single dimension case
    if n_dims == 1:
        axes = [axes]
    
    # Plot residuals for each dimension
    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, residuals[:, i], 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Set labels
        if labels:
            ax.set_ylabel(f"Residual ({labels[i]})", fontsize=11)
        else:
            ax.set_ylabel(f"Residual (y{i})", fontsize=11)
        
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle("Prediction Residuals", fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    return fig, axes


def plot_solution(t, y, labels=None, title="ODE Solution", figsize=(10, 6)):
    """
    Plot ODE solution trajectories.
    
    Parameters
    ----------
    t : array-like
        Time points
    y : ndarray
        Solution at time points (shape: (n_vars, n_times))
    labels : list, optional
        Labels for each solution component
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle 1D or multi-D solutions
    if y.ndim == 1:
        ax.plot(t, y)
    else:
        for i in range(y.shape[0]):
            label = labels[i] if labels else f"y{i}"
            ax.plot(t, y[i], label=label)
        ax.legend()
    
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax
