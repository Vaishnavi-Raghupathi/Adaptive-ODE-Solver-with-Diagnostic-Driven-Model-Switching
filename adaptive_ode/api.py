"""Small supported API; the legacy PINN/Neural ODE modules are experimental."""

import numpy as np

from adaptive_ode.core import BenchmarkConfig, evaluate
from adaptive_ode.oscillator import OscillatorParameters, analytical_trajectory
from adaptive_ode.methods import (
    classical_candidate, learned_polynomial_candidate, oscillator_calibration_candidate, spline_candidate,
    persistence_candidate, linear_trend_candidate,
    classical_suite,
)
from adaptive_ode.numerics import integrate, time_grid, trajectory
from adaptive_ode.systems import DynamicalSystem, get_system


def benchmark(system="mismatch", *, config=None, candidates=None):
    """Compare methods on a built-in key or a supplied DynamicalSystem.

    Known-equation references are generated using the true RHS, while candidate
    integrators use candidate_rhs. The benchmark runs candidates; it does not
    predict a winner without evaluation or promise that ML is beneficial.
    """
    config = config or BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    system = get_system(system) if isinstance(system, str) else system
    if not isinstance(system, DynamicalSystem):
        raise TypeError("system must be a built-in key or DynamicalSystem.")
    t = time_grid(np.linspace(*system.t_span, config.num_points))
    if system is get_system("mismatch"):
        reference = analytical_trajectory(t, system.y0, OscillatorParameters(0.1, 4))
        reference_kind = "analytical reference"
    else:
        reference = integrate(system.rhs, t, system.y0, method=system.reference_method,
                              rtol=1e-11, atol=1e-13, max_nfev=1000000)
        reference_kind = f"numerical reference ({system.reference_method})"
    if candidates is None:
        candidates = classical_suite(system.candidate_rhs, y0=system.y0, max_nfev=config.max_nfev,
                                     sweep=config.solver_sweep, base_step=float(np.median(np.diff(t[:int(len(t)*config.train_fraction)]))))
        if system is get_system("mismatch"):
            candidates.append(oscillator_calibration_candidate(max_nfev=config.max_nfev))
        if config.include_ml:
            from adaptive_ode.ml_methods import trajectory_residual_candidate
            candidates += [trajectory_residual_candidate(system.candidate_rhs, seed=config.seed,
                           alpha=alpha, max_iter=config.nn_max_iter, max_nfev=config.max_nfev)
                           for alpha in [1e-4, 1e-2]]
    result = evaluate(t, reference, candidates, config=config, reference=reference, name=system.name)
    result.reference_kind = reference_kind
    result.context = {"input_kind": "known_equations", "system_key": system.key,
                      "state_labels": list(system.labels),
                      "equation_source": "Retain the DynamicalSystem/RHS source to reproduce custom equations."}
    return result


def benchmark_data(t, observed, *, config=None, candidates=None, rhs=None):
    """Chronological evaluation on uploaded observations; equations not required.

    Test observations are the target, not guaranteed noise-free ground truth.
    Default methods extrapolate beyond their training segment. Supply rhs(t, y)
    to additionally compare numerical integrators against the same observations.
    Equations must use the supplied time/state units and state-column order.
    """
    config = config or BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    t = time_grid(t)
    observed = trajectory(observed, len(t))
    if rhs is not None and not callable(rhs):
        raise TypeError("rhs must be a callable rhs(t, state).")
    if rhs is not None and candidates is not None:
        raise ValueError("Use rhs for default numerical candidates, or supply explicit candidates, not both.")
    if candidates is None:
        candidates = [persistence_candidate(), linear_trend_candidate(), spline_candidate(),
                      learned_polynomial_candidate(degree=1, max_nfev=config.max_nfev),
                      learned_polynomial_candidate(max_nfev=config.max_nfev)]
        if rhs is not None:
            candidates = classical_suite(rhs, max_nfev=config.max_nfev, sweep=config.solver_sweep,
                                         base_step=float(np.median(np.diff(t[:int(len(t)*config.train_fraction)])))) + candidates
        if config.include_ml:
            from adaptive_ode.ml_methods import mlp_trajectory_candidate
            candidates.append(mlp_trajectory_candidate(seed=config.seed, max_iter=config.nn_max_iter))
    result = evaluate(t, observed, candidates, config=config, name="Observed time-series benchmark")
    result.context = {"input_kind": "observations_with_equations" if rhs is not None else "observations",
                      "equation_source": "Python RHS callback; retain its source." if rhs is not None else None}
    return result
