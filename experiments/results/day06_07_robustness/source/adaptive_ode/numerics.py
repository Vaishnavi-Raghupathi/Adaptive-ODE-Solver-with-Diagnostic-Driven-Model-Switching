"""Small numerical primitives used by the public API (NumPy/SciPy only)."""

import numpy as np
from scipy.integrate import solve_ivp


def time_grid(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be a finite, strictly increasing 1D array with at least two points.")
    return t


def trajectory(y, n):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != n or y.shape[1] == 0 or not np.all(np.isfinite(y)):
        raise ValueError("States must be finite with shape (n_times, n_states).")
    return y


def integrate(rhs, t, y0, *, method="DOP853", rtol=1e-7, atol=1e-9, max_nfev=50000, rk4_max_step=None):
    """Integrate from t[0], recording budget exhaustion as an explicit failure."""
    t = time_grid(t)
    y0 = np.asarray(y0, dtype=float)
    if y0.ndim != 1 or len(y0) == 0 or not np.all(np.isfinite(y0)):
        raise ValueError("Initial state must be a nonempty finite vector.")
    evaluations = 0

    def checked_rhs(time, state):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_nfev:
            raise RuntimeError(f"RHS evaluation budget exceeded ({max_nfev}).")
        value = np.asarray(rhs(time, state), dtype=float)
        if value.shape != y0.shape or not np.all(np.isfinite(value)):
            raise ValueError("RHS returned the wrong shape or non-finite derivatives.")
        return value

    if method == "RK4":
        if rk4_max_step is not None and (not np.isfinite(rk4_max_step) or rk4_max_step <= 0):
            raise ValueError("rk4_max_step must be positive and finite.")
        result = np.empty((len(t), len(y0)))
        result[0] = y0
        with np.errstate(over="raise", invalid="raise"):
            for index, interval in enumerate(np.diff(t)):
                time, state = t[index], result[index]
                count = max(1, int(np.ceil(interval/rk4_max_step - 1e-10))) if rk4_max_step else 1
                step = interval/count
                for substep in range(count):
                    current = time + substep*step
                    k1 = checked_rhs(current, state)
                    k2 = checked_rhs(current + step / 2, state + step * k1 / 2)
                    k3 = checked_rhs(current + step / 2, state + step * k2 / 2)
                    k4 = checked_rhs(current + step, state + step * k3)
                    state = state + step * (k1 + 2*k2 + 2*k3 + k4) / 6
                result[index + 1] = state
        return trajectory(result, len(t))
    solution = solve_ivp(checked_rhs, (t[0], t[-1]), y0, t_eval=t, method=method, rtol=rtol, atol=atol)
    if not solution.success or solution.y.shape != (len(y0), len(t)):
        raise RuntimeError(f"Integration failed: {solution.message}")
    return trajectory(solution.y.T, len(t))
