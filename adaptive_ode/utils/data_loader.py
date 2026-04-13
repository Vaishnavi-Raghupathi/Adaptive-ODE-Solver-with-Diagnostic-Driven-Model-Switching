"""Data loading and preprocessing utilities."""

import numpy as np
from scipy.integrate import solve_ivp


def _osc_true(t, s):
    """True damped oscillator: y'' + 0.1y' + 4y = 0."""
    y, v = s
    return [v, -0.1 * v - 4.0 * y]


def _osc_classical(t, s):
    """Wrong oscillator assumed by classical solver: y'' + 0.5y' + 3y = 0."""
    y, v = s
    return [v, -0.5 * v - 3.0 * y]


def generate_lorenz_data(
    t_span=(0, 10),
    y0=None,
    num_points=300,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    noise_std=0.0,
):
    """
    Generate clean or noisy Lorenz trajectory.

    Returns
    -------
    t : ndarray  shape (num_points,)
    y : ndarray  shape (num_points, 3)
    """
    if y0 is None:
        y0 = [1.0, 1.0, 1.0]

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(lorenz, t_span, y0, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-10)
    y = sol.y.T.copy()
    if noise_std > 0.0:
        y += np.random.normal(0.0, noise_std, size=y.shape)
    return sol.t, y


def generate_mismatch_data(
    t_span=(0, 20),
    y0=None,
    num_points=300,
    noise_std=0.0,
):
    """
    Generate damped oscillator trajectory using TRUE parameters.

    True system:      y'' + 0.1y' + 4.0y = 0   (damping=0.1, freq^2=4.0)
    Classical solver: y'' + 0.5y' + 3.0y = 0   (wrong params)

    Returns
    -------
    t        : ndarray  shape (num_points,)
    y        : ndarray  shape (num_points, 2)
    y0_clean : ndarray  shape (2,)
    """
    if y0 is None:
        y0 = [2.0, 0.0]

    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(_osc_true, t_span, y0, t_eval=t_eval,
                    method="RK45", rtol=1e-10, atol=1e-11)
    y = sol.y.T.copy()
    if noise_std > 0.0:
        y += np.random.normal(0.0, noise_std, size=y.shape)
    return sol.t, y, np.array(y0, dtype=np.float32)


def get_classical_mismatch_rhs():
    """Return wrong oscillator RHS for classical solver in mismatch scenario."""
    return _osc_classical


def get_true_mismatch_rhs():
    """Return true oscillator RHS for Neural ODE training target."""
    return _osc_true


def load_data(filepath, delimiter=","):
    return np.loadtxt(filepath, delimiter=delimiter)


def normalize_data(data, axis=0):
    mean = np.mean(data, axis=axis, keepdims=True)
    std  = np.std(data,  axis=axis, keepdims=True)
    return (data - mean) / (std + 1e-8)