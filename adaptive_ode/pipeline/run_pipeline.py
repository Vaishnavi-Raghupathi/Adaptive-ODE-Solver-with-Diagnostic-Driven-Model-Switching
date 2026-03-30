"""Main pipeline orchestration."""

import os
import numpy as np
from adaptive_ode.utils.data_loader import generate_lorenz_data
from adaptive_ode.solvers.classical import ClassicalSolver
from adaptive_ode.solvers.neural_ode import NeuralODESolver
from adaptive_ode.diagnostics.engine import DiagnosticEngine
from adaptive_ode.decision.rules import decide_model
from adaptive_ode.evaluation.metrics import compute_metrics
from adaptive_ode.evaluation.plotting import (
    plot_trajectory,
    plot_residuals,
    plot_model_comparison,
)
import matplotlib.pyplot as plt


def run_pipeline(t_span=(0, 10), num_points=300, noise_std=0.5, y0=None):
    """
    Execute the full ODE solving pipeline with diagnostics and adaptive decisions.
    
    Steps:
    1. Generate Lorenz system data
    2. Initialize and fit classical solver
    3. Compute predictions and residuals
    4. Run diagnostic engine
    5. Make adaptive model selection decision
    6. Plot and report results
    
    Parameters
    ----------
    t_span : tuple
        Time interval (default: (0, 10))
    num_points : int
        Number of time points (default: 300)
    noise_std : float
        Standard deviation of noise (default: 0.5)
    y0 : array-like, optional
        Initial conditions. If None, uses [1.0, 1.0, 1.0]
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 't': time points
        - 'y_true': true trajectories
        - 'y_pred': predicted trajectories
        - 'residuals': prediction residuals
        - 'diagnostics': diagnostic test results
        - 'decision': model selection decision
    """
    
    # Set default initial condition
    if y0 is None:
        y0 = np.array([1.0, 1.0, 1.0])

    print("Running noisy experiment...")
    
    # Step 1: Generate Lorenz data
    print("Generating data...")
    t, y_true = generate_lorenz_data(
        t_span=t_span,
        y0=y0,
        num_points=num_points,
        noise_std=noise_std
    )
    print(f"  Generated {num_points} time points from t={t_span[0]} to t={t_span[1]}")
    print(f"  Data shape: {y_true.shape}")
    
    # Step 2: Define Lorenz ODE function
    def lorenz_ode(t, state):
        """Lorenz system ODE."""
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    # Step 3: Initialize and fit solver
    print("\nRunning classical solver...")
    solver = ClassicalSolver(lorenz_ode)
    solver.fit(t, y_true)
    print("  Solver fitted successfully")
    
    # Step 4: Get predictions
    y_pred = solver.predict()
    print(f"  Predictions shape: {y_pred.shape}")
    
    # Step 5: Compute residuals
    print("\nComputing residuals...")
    residuals = solver.compute_residuals(y_true)
    residual_norm = np.linalg.norm(residuals)
    residual_mean = np.mean(np.abs(residuals))
    print(f"  Residual L2 norm: {residual_norm:.6f}")
    print(f"  Mean absolute residual: {residual_mean:.6f}")
    
    # Step 6: Run diagnostics
    print("\nRunning diagnostics...")
    diagnostic_engine = DiagnosticEngine(y_true, y_pred)
    test_results = diagnostic_engine.run_all_tests()
    
    # Print diagnostics in clean format
    for test_name, result in test_results.items():
        print(f"- {test_name}: {result}")
    
    # Step 7: Make adaptive decision
    print("\nDecision:")
    classical_mse_for_decision = float(np.mean((y_true - y_pred) ** 2))
    model_decision = decide_model(test_results, classical_mse_for_decision)
    print(f"  Selected model: {model_decision}")
    
    # Step 8: Conditionally train/evaluate Neural ODE
    neural_y_pred = None
    if model_decision == "neural_ode":
        print("\nTraining Neural ODE...")
        neural_solver = NeuralODESolver(
            state_dim=y_true.shape[1],
            hidden_dim=64,
            lr=0.001,
            epochs=600,
            print_every=25,
            device='cpu'
        )
        neural_solver.fit(t, y_true)

        print("Evaluating Neural ODE...")
        neural_y_pred = neural_solver.predict()
        print(f"  Neural ODE predictions shape: {neural_y_pred.shape}")

    # Step 9: Compute final metrics
    classical_metrics = compute_metrics(y_true, y_pred)

    neural_metrics = None
    improvement_percent = None
    if neural_y_pred is not None:
        neural_metrics = compute_metrics(y_true, neural_y_pred)
        if classical_metrics["mse"] > 0:
            improvement_percent = (
                (classical_metrics["mse"] - neural_metrics["mse"])
                / classical_metrics["mse"]
            ) * 100.0
        else:
            improvement_percent = 0.0

    print("\n===== FINAL RESULTS =====")
    print("\nClassical Model:")
    print(f"  MSE: {classical_metrics['mse']:.6f}")
    print(f"  RMSE: {classical_metrics['rmse']:.6f}")
    print(f"  MAE: {classical_metrics['mae']:.6f}")

    print("\nNeural ODE:")
    if neural_metrics is not None:
        print(f"  MSE: {neural_metrics['mse']:.6f}")
        print(f"  RMSE: {neural_metrics['rmse']:.6f}")
        print(f"  MAE: {neural_metrics['mae']:.6f}")
    else:
        print("  MSE: N/A")
        print("  RMSE: N/A")
        print("  MAE: N/A")

    print("\nImprovement:")
    if improvement_percent is not None:
        print(f"  % improvement: {improvement_percent:.2f}%")
    else:
        print("  % improvement: N/A")

    # Step 10: Plot results
    print("\nGenerating plots...")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trajectory comparison
    fig1, axes1 = plot_trajectory(
        t, y_true, y_pred,
        labels=['x', 'y', 'z']
    )
    trajectory_path = os.path.join(output_dir, 'trajectory_comparison.png')
    fig1.savefig(trajectory_path, dpi=100, bbox_inches='tight')
    print(f"  Saved: {trajectory_path}")
    
    # Plot residuals
    fig2, axes2 = plot_residuals(
        t, residuals,
        labels=['x', 'y', 'z']
    )
    residuals_path = os.path.join(output_dir, 'residuals.png')
    fig2.savefig(residuals_path, dpi=100, bbox_inches='tight')
    print(f"  Saved: {residuals_path}")

    # Plot model comparison
    if neural_y_pred is not None:
        fig3, axes3 = plot_model_comparison(t, y_true, y_pred, neural_y_pred)
        model_comparison_path = os.path.join(output_dir, 'model_comparison.png')
        fig3.savefig(model_comparison_path, dpi=100, bbox_inches='tight')
        print(f"  Saved: {model_comparison_path}")
    else:
        print("  Skipped: model_comparison.png (Neural ODE not selected)")
    
    plt.show()
    
    # Return results
    results = {
        't': t,
        'y_true': y_true,
        'y_pred': y_pred,
        'classical_y_pred': y_pred,
        'neural_y_pred': neural_y_pred,
        'classical_metrics': classical_metrics,
        'neural_metrics': neural_metrics,
        'improvement_percent': improvement_percent,
        'residuals': residuals,
        'diagnostics': {
            'test_results': test_results,
            'error_metrics': diagnostic_engine.get_error_metrics()
        },
        'decision': model_decision
    }
    
    print("\n✓ Pipeline completed successfully!")
    return results


if __name__ == "__main__":
    # Run the pipeline with default parameters
    results = run_pipeline()
