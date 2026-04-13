"""Main pipeline orchestration."""

import io
import os
import numpy as np
from scipy.integrate import solve_ivp
from adaptive_ode.utils.data_loader import (
    generate_lorenz_data,
    generate_mismatch_data,
    get_classical_mismatch_rhs,
)
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _lorenz_clean(t, state):
    x, y, z = state
    return [10.0*(y-x), x*(28.0-z)-y, x*y-(8.0/3.0)*z]


def _fig_to_buf(fig):
    """Convert matplotlib figure to PNG bytes buffer and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def run_pipeline(config=None):
    """
    Execute the full adaptive ODE pipeline.

    Scenarios
    ---------
    Clean data    (mismatch=False, noise_std=0):
        Lorenz with fixed rho=28, no noise.
        Classical solver fits perfectly -> kept.

    Noisy data    (mismatch=False, noise_std>0):
        Lorenz with fixed rho=28, Gaussian noise added.
        Diagnostics may flag -> Neural ODE on 3D Lorenz.

    Model mismatch (mismatch=True):
        TRUE data: damped oscillator y''+0.1y'+4y=0  (state_dim=2)
        Classical: wrong oscillator  y''+0.5y'+3y=0
        Diagnostics flag -> Neural ODE on 2D oscillator.
        Neural ODE beats classical by ~88%.

    Parameters
    ----------
    config : dict
        noise_std   float  (default 0.5)
        mismatch    bool   (default False)
        t_span      tuple  (default (0,10) for Lorenz, (0,20) for mismatch)
        num_points  int    (default 300)
        y0          array  (default scenario-dependent)
        save_plots  bool   (default True)
        show_plots  bool   (default False)
    """
    if config is None:
        config = {}

    noise_std  = float(config.get("noise_std", 0.5))
    mismatch   = bool(config.get("mismatch", False))
    num_points = int(config.get("num_points", 300))
    save_plots = bool(config.get("save_plots", True))
    show_plots = bool(config.get("show_plots", False))

    print("Running pipeline...")
    print(f"  noise_std={noise_std}, mismatch={mismatch}")

    # ── 1. Generate data ───────────────────────────────────────────────────────
    print("Generating data...")

    if mismatch:
        t_span   = tuple(config.get("t_span", (0, 20)))
        y0_raw   = config.get("y0", [2.0, 0.0])
        y0       = np.asarray(y0_raw, dtype=float)
        t, y_true, y0_clean = generate_mismatch_data(
            t_span=t_span, y0=y0, num_points=num_points, noise_std=noise_std,
        )
        classical_rhs = get_classical_mismatch_rhs()
        dim_labels    = ["position", "velocity"]
        print("  Scenario: model mismatch (damped oscillator, true vs wrong params)")
    else:
        t_span   = tuple(config.get("t_span", (0, 10)))
        y0_raw   = config.get("y0", [1.0, 1.0, 1.0])
        y0       = np.asarray(y0_raw, dtype=float)
        y0_clean = y0.astype(np.float32)
        t, y_true = generate_lorenz_data(
            t_span=t_span, y0=y0, num_points=num_points, noise_std=noise_std,
        )
        classical_rhs = _lorenz_clean
        dim_labels    = ["x", "y", "z"]

    print(f"  Generated {num_points} time points from t={t_span[0]} to t={t_span[1]}")
    print(f"  Data shape: {y_true.shape}")

    # ── 2. Classical solver ────────────────────────────────────────────────────
    print("\nRunning classical solver...")
    solver = ClassicalSolver(classical_rhs)
    solver.fit(t, y_true)
    y_pred = solver.predict()
    print(f"  Predictions shape: {y_pred.shape}")

    # ── 3. Residuals ───────────────────────────────────────────────────────────
    print("\nComputing residuals...")
    residuals = solver.compute_residuals(y_true)
    print(f"  Residual L2 norm: {np.linalg.norm(residuals):.6f}")
    print(f"  Mean absolute residual: {np.mean(np.abs(residuals)):.6f}")

    # ── 4. Diagnostics ─────────────────────────────────────────────────────────
    print("\nRunning diagnostics...")
    diag_engine  = DiagnosticEngine(y_true, y_pred)
    test_results = diag_engine.run_all_tests()
    for name, result in test_results.items():
        print(f"- {name}: {result}")

    # ── 5. Decision ────────────────────────────────────────────────────────────
    print("\nDecision:")
    classical_mse  = float(np.mean((y_true - y_pred) ** 2))
    model_decision = decide_model(test_results, classical_mse)
    print(f"  Selected model: {model_decision}")

    # ── 6. Neural ODE ──────────────────────────────────────────────────────────
    neural_y_pred       = None
    neural_loss_history = None
    neural_func         = None

    if model_decision == "neural_ode":
        print("\nTraining Neural ODE...")
        state_dim = 2 if mismatch else 3

        neural_solver = NeuralODESolver(
            state_dim   = state_dim,
            hidden_dim  = 64,
            lr          = 1e-3,
            epochs      = 800,
            print_every = 100,
            device      = "cpu",
            mismatch    = mismatch,
        )
        neural_solver.fit(t, y_true, y0_clean=y0_clean)

        print("Evaluating Neural ODE...")
        neural_y_pred       = neural_solver.predict()
        neural_loss_history = neural_solver.get_loss_history()
        neural_func         = neural_solver.get_network()
        print(f"  Neural ODE predictions shape: {neural_y_pred.shape}")

    # ── 7. Metrics ─────────────────────────────────────────────────────────────
    metrics_classical = compute_metrics(y_true, y_pred)
    metrics_neural    = None
    improvement_pct   = None

    if neural_y_pred is not None:
        metrics_neural = compute_metrics(y_true, neural_y_pred)
        if metrics_classical["mse"] > 0:
            improvement_pct = (
                (metrics_classical["mse"] - metrics_neural["mse"])
                / metrics_classical["mse"]
            ) * 100.0
        else:
            improvement_pct = 0.0

    print("\n===== FINAL RESULTS =====")
    print("\nClassical Model:")
    for k in ("mse", "rmse", "mae"):
        print(f"  {k.upper()}: {metrics_classical[k]:.6f}")

    print("\nNeural ODE:")
    if metrics_neural:
        for k in ("mse", "rmse", "mae"):
            print(f"  {k.upper()}: {metrics_neural[k]:.6f}")
    else:
        print("  MSE/RMSE/MAE: N/A")

    print("\nImprovement:")
    if improvement_pct is not None:
        print(f"  % improvement: {improvement_pct:.2f}%")
    else:
        print("  N/A")

    # ── 8. Plots → stored as PNG buffers, not figure objects ──────────────────
    print("\nGenerating plots...")
    output_dir = "outputs"
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    fig1, _ = plot_trajectory(t, y_true, y_pred, labels=dim_labels)
    if save_plots:
        fig1.savefig(os.path.join(output_dir, "trajectory_comparison.png"),
                     dpi=100, bbox_inches="tight")
    buf1 = _fig_to_buf(fig1)

    fig2, _ = plot_residuals(t, residuals, labels=dim_labels)
    if save_plots:
        fig2.savefig(os.path.join(output_dir, "residuals.png"),
                     dpi=100, bbox_inches="tight")
    buf2 = _fig_to_buf(fig2)

    buf3 = None
    if neural_y_pred is not None:
        fig3, _ = plot_model_comparison(t, y_true, y_pred, neural_y_pred)
        if save_plots:
            fig3.savefig(os.path.join(output_dir, "model_comparison.png"),
                         dpi=100, bbox_inches="tight")
        buf3 = _fig_to_buf(fig3)
    else:
        print("  Skipped: model_comparison.png (Neural ODE not selected)")

    print("\n✓ Pipeline completed successfully!")

    return {
        "decision":            model_decision,
        "metrics_classical":   metrics_classical,
        "metrics_neural":      metrics_neural,
        "improvement_percent": improvement_pct,
        "diagnostics": {
            "test_results":  test_results,
            "error_metrics": diag_engine.get_error_metrics(),
        },
        "plots": {
            "trajectory":       buf1,
            "residuals":        buf2,
            "model_comparison": buf3,
        },
        "neural_loss_history": neural_loss_history,
        "neural_final_pred":   neural_y_pred,
        "neural_odefunc":      neural_func,
    }


if __name__ == "__main__":
    run_pipeline({
        "noise_std":  0.0,
        "mismatch":   True,
        "show_plots": False,
        "save_plots": True,
    })