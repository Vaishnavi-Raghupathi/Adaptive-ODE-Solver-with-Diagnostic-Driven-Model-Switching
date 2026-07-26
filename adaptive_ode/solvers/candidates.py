"""Solver candidates used by the adaptive selection pipeline."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import warnings

from adaptive_ode.evaluation.metrics import compute_metrics


@dataclass
class SolverResult:
    name: str
    family: str
    prediction: np.ndarray | None
    metrics: dict | None
    runtime_seconds: float
    status: str
    message: str


def solve_reference(system, t):
    sol = solve_ivp(
        system.rhs,
        (float(t[0]), float(t[-1])),
        system.y0,
        t_eval=t,
        method=system.reference_method,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Reference solve failed: {sol.message}")
    return sol.y.T


def solve_rk4(rhs, t, y0):
    t = np.asarray(t, dtype=float)
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = np.asarray(y0, dtype=float)

    for idx in range(len(t) - 1):
        step = float(t[idx + 1] - t[idx])
        current_t = float(t[idx])
        current_y = y[idx]
        k1 = np.asarray(rhs(current_t, current_y), dtype=float)
        k2 = np.asarray(rhs(current_t + step / 2.0, current_y + step * k1 / 2.0), dtype=float)
        k3 = np.asarray(rhs(current_t + step / 2.0, current_y + step * k2 / 2.0), dtype=float)
        k4 = np.asarray(rhs(current_t + step, current_y + step * k3), dtype=float)
        y[idx + 1] = current_y + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y


def _test_metrics(y_reference, prediction, test_mask):
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction contains non-finite values.")
    return compute_metrics(y_reference[test_mask], prediction[test_mask])


def run_rk4(system, t, y_reference, test_mask):
    start = perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            prediction = solve_rk4(system.candidate_rhs, t, system.y0)
        return SolverResult(
            name="RK4",
            family="Classical explicit",
            prediction=prediction,
            metrics=_test_metrics(y_reference, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message="Fixed-step fourth-order Runge-Kutta on the evaluation grid.",
        )
    except Exception as exc:
        return SolverResult("RK4", "Classical explicit", None, None, perf_counter() - start, "failed", str(exc))


def run_scipy_method(system, method, t, y_reference, test_mask):
    start = perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = solve_ivp(
                system.candidate_rhs,
                (float(t[0]), float(t[-1])),
                system.y0,
                t_eval=t,
                method=method,
                rtol=1e-6,
                atol=1e-8,
            )
        if not sol.success:
            raise RuntimeError(sol.message)
        prediction = sol.y.T
        family = "Classical implicit" if method in {"BDF", "Radau", "LSODA"} else "Classical explicit"
        return SolverResult(
            name=method,
            family=family,
            prediction=prediction,
            metrics=_test_metrics(y_reference, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message=f"SciPy {method} solve with moderate tolerances.",
        )
    except Exception as exc:
        return SolverResult(method, "Classical", None, None, perf_counter() - start, "failed", str(exc))


def run_hybrid_residual(system, t, y_observed, y_reference, train_mask, test_mask):
    start = perf_counter()
    name = "RK4 + residual NN"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            baseline = solve_rk4(system.candidate_rhs, t, system.y0)
        residual = y_observed - baseline
        features = np.column_stack([t, baseline])

        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="tanh",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1200,
                random_state=7,
            ),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(features[train_mask], residual[train_mask])

        prediction = baseline + model.predict(features)
        return SolverResult(
            name=name,
            family="Hybrid correction",
            prediction=prediction,
            metrics=_test_metrics(y_reference, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message="Learns held-out residual correction from observed training trajectory.",
        )
    except Exception as exc:
        return SolverResult(name, "Hybrid correction", None, None, perf_counter() - start, "failed", str(exc))


def run_pinn_surrogate(system, t, y_observed, y_reference, train_mask, test_mask, epochs=250):
    start = perf_counter()
    name = "PINN surrogate"

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        return SolverResult(name, "Physics-informed NN", None, None, perf_counter() - start, "skipped", str(exc))

    try:
        t_np = np.asarray(t, dtype=np.float32)
        y_np = np.asarray(y_observed, dtype=np.float32)
        state_dim = y_np.shape[1]

        t_min = float(t_np[0])
        t_scale = float(t_np[-1] - t_np[0])
        y_mean = y_np[train_mask].mean(axis=0)
        y_std = y_np[train_mask].std(axis=0) + 1e-6

        def normalize_t(values):
            return ((values - t_min) / t_scale).reshape(-1, 1)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, state_dim),
                )

            def forward(self, values):
                return self.net(values)

        def torch_rhs(y_values):
            if system.key == "lorenz":
                x = y_values[:, 0]
                y = y_values[:, 1]
                z = y_values[:, 2]
                return torch.stack([
                    10.0 * (y - x),
                    x * (28.0 - z) - y,
                    x * y - (8.0 / 3.0) * z,
                ], dim=1)
            if system.key == "vanderpol":
                y = y_values[:, 0]
                velocity = y_values[:, 1]
                return torch.stack([
                    velocity,
                    5.0 * (1.0 - y**2) * velocity - y,
                ], dim=1)
            if system.key == "robertson":
                y1 = y_values[:, 0]
                y2 = y_values[:, 1]
                y3 = y_values[:, 2]
                return torch.stack([
                    -0.04 * y1 + 1.0e4 * y2 * y3,
                    0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
                    3.0e7 * y2 * y2,
                ], dim=1)
            y = y_values[:, 0]
            velocity = y_values[:, 1]
            return torch.stack([velocity, -0.1 * velocity - 4.0 * y], dim=1)

        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        t_train = torch.tensor(normalize_t(t_np[train_mask]), dtype=torch.float32)
        y_train = torch.tensor((y_np[train_mask] - y_mean) / y_std, dtype=torch.float32)
        t_phys = torch.tensor(normalize_t(t_np), dtype=torch.float32, requires_grad=True)

        for _ in range(int(epochs)):
            optimizer.zero_grad()
            data_pred = model(t_train)
            data_loss = F.mse_loss(data_pred, y_train)

            phys_pred_norm = model(t_phys)
            phys_pred = phys_pred_norm * torch.tensor(y_std, dtype=torch.float32) + torch.tensor(y_mean, dtype=torch.float32)

            dy_dt = []
            for dim in range(state_dim):
                grad = torch.autograd.grad(
                    phys_pred[:, dim].sum(),
                    t_phys,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, 0] / t_scale
                dy_dt.append(grad)
            dy_dt = torch.stack(dy_dt, dim=1)

            rhs_pred = torch_rhs(phys_pred)
            rhs_scale = torch.clamp(rhs_pred.detach().abs().mean(dim=0), min=1.0)
            physics_loss = F.mse_loss(dy_dt / rhs_scale, rhs_pred / rhs_scale)
            loss = data_loss + 1e-3 * physics_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            all_t = torch.tensor(normalize_t(t_np), dtype=torch.float32)
            pred_norm = model(all_t).numpy()
            prediction = pred_norm * y_std + y_mean

        return SolverResult(
            name=name,
            family="Physics-informed NN",
            prediction=prediction,
            metrics=_test_metrics(y_reference, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message="Fits trajectory data while penalizing ODE residual mismatch.",
        )
    except Exception as exc:
        return SolverResult(name, "Physics-informed NN", None, None, perf_counter() - start, "failed", str(exc))


def choose_best(results):
    successful = [
        result
        for result in results
        if (
            result.status == "ok"
            and result.metrics is not None
            and np.isfinite(result.metrics["mse"])
        )
    ]
    if not successful:
        return None

    return min(successful, key=lambda result: result.metrics["mse"])


def run_cubic_spline_data(t, y_observed, train_mask, test_mask):
    start = perf_counter()
    name = "Cubic spline"
    try:
        train_idx = np.where(train_mask)[0]
        order = np.argsort(t[train_idx])
        t_train = t[train_idx][order]
        y_train = y_observed[train_idx][order]
        model = CubicSpline(t_train, y_train, axis=0, extrapolate=True)
        prediction = model(t)
        return SolverResult(
            name=name,
            family="Data interpolation",
            prediction=prediction,
            metrics=_test_metrics(y_observed, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message="Fits a smooth trajectory directly from uploaded data.",
        )
    except Exception as exc:
        return SolverResult(name, "Data interpolation", None, None, perf_counter() - start, "failed", str(exc))


def run_mlp_trajectory_data(t, y_observed, train_mask, test_mask):
    start = perf_counter()
    name = "MLP trajectory"
    try:
        features = np.asarray(t, dtype=float).reshape(-1, 1)
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="tanh",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1500,
                random_state=11,
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(features[train_mask], y_observed[train_mask])
        prediction = model.predict(features)
        return SolverResult(
            name=name,
            family="Data-driven NN",
            prediction=prediction,
            metrics=_test_metrics(y_observed, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message="Learns a direct neural map from time to state.",
        )
    except Exception as exc:
        return SolverResult(name, "Data-driven NN", None, None, perf_counter() - start, "failed", str(exc))


def run_sindy_polynomial_data(t, y_observed, train_mask, test_mask):
    start = perf_counter()
    name = "Polynomial learned ODE"
    try:
        train_idx = np.where(train_mask)[0]
        order = np.argsort(t[train_idx])
        t_train = np.asarray(t[train_idx][order], dtype=float)
        y_train = np.asarray(y_observed[train_idx][order], dtype=float)
        if len(t_train) < 8:
            raise ValueError("At least 8 training points are required for learned ODE fitting.")

        dydt = np.gradient(y_train, t_train, axis=0)
        library = PolynomialFeatures(degree=2, include_bias=True)
        features = library.fit_transform(y_train)
        model = Ridge(alpha=1e-5)
        model.fit(features, dydt)

        def learned_rhs(eval_t, state):
            del eval_t
            state = np.asarray(state, dtype=float).reshape(1, -1)
            return model.predict(library.transform(state))[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = solve_ivp(
                learned_rhs,
                (float(t[0]), float(t[-1])),
                y_observed[0],
                t_eval=t,
                method="RK45",
                rtol=1e-5,
                atol=1e-7,
            )
        if not sol.success:
            raise RuntimeError(sol.message)
        prediction = sol.y.T
        return SolverResult(
            name=name,
            family="Data-driven dynamics",
            prediction=prediction,
            metrics=_test_metrics(y_observed, prediction, test_mask),
            runtime_seconds=perf_counter() - start,
            status="ok",
            message="Discovers a low-order polynomial state derivative and integrates it forward.",
        )
    except Exception as exc:
        return SolverResult(name, "Data-driven dynamics", None, None, perf_counter() - start, "failed", str(exc))
