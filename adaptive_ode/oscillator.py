"""Oscillator parameters, analytical reference, and training-only calibration."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class OscillatorParameters:
    damping: float
    stiffness: float

    def __post_init__(self):
        if not np.isfinite(self.damping) or self.damping < 0:
            raise ValueError("Damping must be finite and nonnegative.")
        if not np.isfinite(self.stiffness) or self.stiffness <= 0:
            raise ValueError("Stiffness must be finite and positive.")

    def rhs(self, time, state):
        del time
        position, velocity = state
        return np.array([velocity, -self.damping * velocity - self.stiffness * position])


def validate_times(t):
    values = np.asarray(t, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Time must be a 1D array with at least two samples.")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError("Time must be finite and strictly increasing.")
    return values


def validate_initial_state(y0):
    values = np.asarray(y0, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("Initial state must contain two finite values.")
    return values


def analytical_trajectory(t, y0, parameters):
    """Closed-form underdamped solution; no numerical solver generates truth.

    The initial state is defined at t[0]. The experiment only uses an
    underdamped true system. Numerical baselines also support other regimes.
    """
    t = validate_times(t)
    position0, velocity0 = validate_initial_state(y0)
    decay = parameters.damping / 2
    frequency_squared = parameters.stiffness - decay**2
    if frequency_squared <= 0:
        raise ValueError("Analytical reference requires an underdamped oscillator.")
    frequency = np.sqrt(frequency_squared)
    elapsed = t - t[0]
    a = position0
    b = (velocity0 + decay * position0) / frequency
    cosine = np.cos(frequency * elapsed)
    sine = np.sin(frequency * elapsed)
    envelope = np.exp(-decay * elapsed)
    position = envelope * (a * cosine + b * sine)
    velocity = envelope * (
        (-decay * a + frequency * b) * cosine
        + (-decay * b - frequency * a) * sine
    )
    return np.column_stack([position, velocity])


def integrate_oscillator(t, y0, parameters, *, rtol=1e-6, atol=1e-8):
    t = validate_times(t)
    y0 = validate_initial_state(y0)
    result = solve_ivp(
        parameters.rhs, (float(t[0]), float(t[-1])), y0,
        t_eval=t, method="DOP853", rtol=rtol, atol=atol,
    )
    if not result.success or result.y.shape != (2, len(t)):
        raise RuntimeError(f"Oscillator integration failed: {result.message}")
    prediction = result.y.T
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Oscillator integration returned non-finite values.")
    return prediction


def fit_parameters(t_train, y_train, y0, initial_parameters, *, max_nfev=200):
    """Fit damping/stiffness using training samples only and a known initial state.

    Both state dimensions are observed and equally weighted in raw units.
    The bounded least-squares optimization has a single prescribed start.
    """
    t_train = validate_times(t_train)
    y0 = validate_initial_state(y0)
    y_train = np.asarray(y_train, dtype=float)
    if y_train.shape != (len(t_train), 2) or not np.all(np.isfinite(y_train)):
        raise ValueError("Training observations must be finite with shape (n, 2).")

    def residual(values):
        parameters = OscillatorParameters(*values)
        predicted = integrate_oscillator(
            t_train, y0, parameters, rtol=1e-9, atol=1e-11,
        )
        return (predicted - y_train).ravel()

    result = least_squares(
        residual,
        [initial_parameters.damping, initial_parameters.stiffness],
        bounds=([0.0, 0.5], [2.0, 8.0]),
        max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    if not result.success:
        raise RuntimeError(f"Parameter fit failed: {result.message}")
    fitted = OscillatorParameters(*map(float, result.x))
    return fitted, {
        "optimizer": "scipy.optimize.least_squares",
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "message": str(result.message),
        "bounds": [[0.0, 0.5], [2.0, 8.0]],
        "active_mask": result.active_mask.tolist(),
    }
