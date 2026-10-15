"""Public NumPy/SciPy candidates; ML candidates live in ml_methods."""

from dataclasses import asdict

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from adaptive_ode.core import Candidate, FittedModel
from adaptive_ode.oscillator import OscillatorParameters, fit_parameters
from adaptive_ode.numerics import integrate


def integrated_model(rhs, t0, y0, *, details=None, method="DOP853", rtol=1e-7, atol=1e-9, max_nfev=50000):
    y0 = np.asarray(y0, dtype=float).copy()

    def predict(t):
        if t[0] != t0:
            raise ValueError("Prediction grid must start at the fitted model's initial time.")
        return integrate(rhs, t, y0, method=method, rtol=rtol, atol=atol, max_nfev=max_nfev)

    return FittedModel(
        predict, details or {},
        lambda t, initial: integrate(rhs, t, initial, method=method, rtol=rtol, atol=atol, max_nfev=max_nfev),
    )


def classical_candidate(rhs, method="DOP853", *, name=None, y0=None, rtol=1e-7, atol=1e-9, max_nfev=50000):
    """Numerical integration of supplied equations, without learning parameters."""
    def fit(t, y):
        return integrated_model(rhs, float(t[0]), y[0] if y0 is None else y0,
                                details={"solver": method, "rtol": rtol, "atol": atol, "max_nfev": max_nfev},
                                method=method, rtol=rtol, atol=atol, max_nfev=max_nfev)
    return Candidate(name or method.lower(), "classical", fit)


def oscillator_calibration_candidate(assumed=None, *, max_nfev=50000):
    """Fit a linear oscillator to both observed states; known model structure."""
    assumed = assumed or OscillatorParameters(0.5, 3.0)

    def fit(t, y):
        if y.shape[1] != 2:
            raise ValueError("Oscillator calibration requires position and velocity columns.")
        parameters, details = fit_parameters(t, y, y[0], assumed)
        return integrated_model(parameters.rhs, float(t[0]), y[0],
                                details={**details, "parameters": asdict(parameters)},
                                rtol=1e-9, atol=1e-11, max_nfev=max_nfev)
    return Candidate("calibrated_oscillator", "calibrated physics", fit)


def smoothed_acceleration(t, y, window):
    """Training-only Savitzky–Golay smoothing; requires a uniform time grid.

    Derivatives are estimated from velocity; no true derivatives are supplied.
    Use a window smaller than the number of retained training observations.
    """
    if y.shape[1] != 2:
        raise ValueError("Acceleration models require position and velocity columns.")
    if isinstance(window, bool) or not isinstance(window, int) or window < 5 or window % 2 == 0 or window > len(t):
        raise ValueError("Smoothing window must be odd, >=5, and no longer than training data.")
    steps = np.diff(t)
    if not np.allclose(steps, steps[0], atol=1e-10, rtol=1e-6):
        raise ValueError("Acceleration correction currently requires uniformly sampled training times.")
    smoothed = savgol_filter(y, window, 3, axis=0)
    acceleration = savgol_filter(y[:, 1], window, 3, deriv=1, delta=float(steps[0]))
    # Boundary derivative estimates are weaker; exclude the filter half-window.
    keep = slice(window // 2, len(t) - window // 2)
    if len(t[keep]) < 5:
        raise ValueError("Too few interior samples remain after smoothing.")
    return smoothed[keep], acceleration[keep]


def cubic_features(states):
    states = np.asarray(states)
    x, v = states[..., 0], states[..., 1]
    return np.stack([x, v, x**3, x*x*v, x*v*v, v**3], axis=-1)


def polynomial_correction_candidate(assumed=None, *, window=9, alpha=1e-6, max_nfev=50000):
    """Learn a cubic acceleration residual on states, then integrate the ODE.

    The fixed library can represent the study's missing cubic restoring force.
    This is an intentionally favorable, interpretable baseline, not discovery
    of arbitrary unknown physics or a sparse SINDy implementation.
    """
    assumed = assumed or OscillatorParameters(0.5, 3.0)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")

    def fit(t, y):
        states, acceleration = smoothed_acceleration(t, y, window)
        target = acceleration + assumed.damping*states[:, 1] + assumed.stiffness*states[:, 0]
        features = cubic_features(states)
        scale = np.maximum(features.std(axis=0), 1e-8)
        design = features / scale
        # An augmented least-squares solve avoids squaring the condition number.
        augmented = np.vstack([design, np.sqrt(alpha) * np.eye(design.shape[1])])
        coefficients = np.linalg.lstsq(augmented, np.concatenate([target, np.zeros(design.shape[1])]), rcond=None)[0] / scale

        def rhs(time, state):
            del time
            x, v = state
            return np.array([v, -assumed.damping*v - assumed.stiffness*x + cubic_features(state) @ coefficients])

        return integrated_model(rhs, float(t[0]), y[0], details={
            "window": window, "alpha": alpha, "coefficients": coefficients.tolist(),
            "feature_names": ["x", "v", "x^3", "x^2*v", "x*v^2", "v^3"],
            "derivative_source": "training velocity, Savitzky-Golay", "assumed": asdict(assumed),
        }, max_nfev=max_nfev)
    return Candidate(f"polynomial_correction_w{window}", "state-based polynomial correction", fit, "polynomial_correction")


def spline_candidate():
    def fit(t, y):
        model = CubicSpline(t, y, axis=0, extrapolate=True)
        return FittedModel(model, {"extrapolate": True, "warning": "Cubic extrapolation can be unstable outside training times."})
    return Candidate("cubic_spline", "trajectory extrapolation", fit)


def learned_polynomial_candidate(*, alpha=1e-5, max_nfev=50000):
    """Fit a degree-two derivative model using finite differences (not sparse SINDy)."""
    def features(y):
        return np.concatenate([np.ones(y.shape[:-1] + (1,)), y,
                               np.stack([y[..., i]*y[..., j] for i in range(y.shape[-1])
                                         for j in range(i, y.shape[-1])], axis=-1)], axis=-1)

    def fit(t, y):
        if len(t) < 8:
            raise ValueError("A learned polynomial ODE requires at least eight training points.")
        derivative = np.gradient(y, t, axis=0)
        design = features(y)
        scale = np.maximum(np.std(design, axis=0), 1e-8)
        scale[0] = 1
        normalized = design / scale
        augmented = np.vstack([normalized, np.sqrt(alpha) * np.eye(normalized.shape[1])])
        target = np.vstack([derivative, np.zeros((normalized.shape[1], y.shape[1]))])
        coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0] / scale[:, None]
        return integrated_model(lambda time, state: features(state) @ coefficients, float(t[0]), y[0],
                                details={"degree": 2, "alpha": alpha, "derivative_source": "training finite differences"},
                                max_nfev=max_nfev)
    return Candidate("polynomial_learned_ode", "data-driven dynamics", fit)
