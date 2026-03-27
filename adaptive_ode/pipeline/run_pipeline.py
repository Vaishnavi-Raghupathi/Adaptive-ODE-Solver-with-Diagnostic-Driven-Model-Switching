"""Main pipeline orchestration."""

import numpy as np
from adaptive_ode.utils.data_loader import generate_lorenz_data
from adaptive_ode.solvers.classical import ClassicalSolver
from adaptive_ode.diagnostics.engine import DiagnosticEngine
from adaptive_ode.decision.rules import decide_model
from adaptive_ode.evaluation.plotting import plot_trajectory, plot_residuals
import matplotlib.pyplot as plt


def run_pipeline(t_span=(0, 50), num_points=1000, noise_std=0.01, y0=None):
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
        Time interval (default: (0, 50))
    num_points : int
        Number of time points (default: 1000)
    noise_std : float
        Standard deviation of noise (default: 0.01)
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
    model_decision = decide_model(test_results)
    print(f"  Selected model: {model_decision}")
    
    # Step 8: Plot results
    print("\nGenerating plots...")
    
    # Plot trajectory comparison
    fig1, axes1 = plot_trajectory(
        t, y_true, y_pred,
        labels=['x', 'y', 'z']
    )
    plt.savefig('trajectory_comparison.png', dpi=100, bbox_inches='tight')
    print("  Saved: trajectory_comparison.png")
    
    # Plot residuals
    fig2, axes2 = plot_residuals(
        t, residuals,
        labels=['x', 'y', 'z']
    )
    plt.savefig('residuals.png', dpi=100, bbox_inches='tight')
    print("  Saved: residuals.png")
    
    plt.show()
    
    # Return results
    results = {
        't': t,
        'y_true': y_true,
        'y_pred': y_pred,
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
