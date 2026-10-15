"""Optional scikit-learn candidates; no PyTorch dependency in the public API."""

import numpy as np

from adaptive_ode.core import Candidate, FittedModel
from adaptive_ode.experiments.oscillator import OscillatorParameters
from adaptive_ode.methods import integrated_model, smoothed_acceleration
from adaptive_ode.numerics import integrate


def _regressor(seed, alpha, max_iter):
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError("ML candidates require: pip install 'adaptive-solver-bench[ml]'") from error
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="tanh", solver="adam", alpha=alpha,
        learning_rate_init=0.001, max_iter=max_iter, random_state=seed,
        early_stopping=False, tol=1e-4, n_iter_no_change=10,
    ))


def _details(model, **extra):
    learner = model.steps[-1][1]
    return {**extra, "n_iter": int(learner.n_iter_), "loss": float(learner.loss_),
            "loss_curve": list(map(float, learner.loss_curve_))}


def trajectory_residual_candidate(rhs, *, seed=7, alpha=1e-4, max_iter=1200, max_nfev=50000):
    def fit(t, y):
        initial, origin = y[0].copy(), float(t[0])
        baseline = integrate(rhs, t, initial, method="RK4", max_nfev=max_nfev, rk4_max_step=0.05)
        model = _regressor(seed, alpha, max_iter)
        model.fit(np.column_stack([t, baseline]), y-baseline)

        def predict(times):
            if times[0] != origin:
                raise ValueError("Prediction times must begin at the original initial time.")
            baseline_all = integrate(rhs, times, initial, method="RK4", max_nfev=max_nfev, rk4_max_step=0.05)
            return baseline_all + model.predict(np.column_stack([times, baseline_all]))
        return FittedModel(predict, _details(model, alpha=alpha, seed=seed, transfer_supported=False))
    return Candidate(f"trajectory_residual_nn_a{alpha:g}", "trajectory residual NN", fit, "trajectory_residual_nn")


def acceleration_residual_candidate(assumed=None, *, seed=7, alpha=1e-4, window=9, max_iter=1200, max_nfev=50000):
    """Learn an acceleration correction f(x,v), supporting new initial states."""
    assumed = assumed or OscillatorParameters(0.5, 3.0)

    def fit(t, y):
        states, acceleration = smoothed_acceleration(t, y, window)
        target = acceleration + assumed.damping*states[:, 1] + assumed.stiffness*states[:, 0]
        mean, scale = float(target.mean()), max(float(target.std()), 1e-8)
        model = _regressor(seed, alpha, max_iter)
        model.fit(states, (target-mean)/scale)

        def rhs(time, state):
            del time
            correction = float(model.predict(np.asarray(state).reshape(1, 2))[0]) * scale + mean
            return np.array([state[1], -assumed.damping*state[1] - assumed.stiffness*state[0] + correction])

        return integrated_model(rhs, float(t[0]), y[0], details=_details(
            model, alpha=alpha, seed=seed, window=window, transfer_supported=True,
            target_mean=mean, target_scale=scale, derivative_source="training velocity, Savitzky-Golay",
        ), max_nfev=max_nfev)
    return Candidate(f"acceleration_residual_nn_a{alpha:g}", "state-based residual NN", fit, "acceleration_residual_nn")


def mlp_trajectory_candidate(*, seed=7, max_iter=1200):
    def fit(t, y):
        model = _regressor(seed, 1e-4, max_iter)
        # Always use a matrix target to preserve the one-state output dimension.
        model.fit(t[:, None], y)
        def predict(times):
            prediction = model.predict(times[:, None])
            return prediction[:, None] if prediction.ndim == 1 else prediction
        return FittedModel(predict, _details(model, seed=seed, transfer_supported=False))
    return Candidate("mlp_trajectory", "data-driven trajectory NN", fit)
