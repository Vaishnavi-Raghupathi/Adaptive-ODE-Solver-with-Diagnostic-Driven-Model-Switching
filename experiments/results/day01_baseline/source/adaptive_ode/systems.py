"""Built-in dynamical systems for solver benchmarking."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayRhs = Callable[[float, np.ndarray], list[float] | np.ndarray]


@dataclass(frozen=True)
class DynamicalSystem:
    key: str
    name: str
    description: str
    stiffness: str
    t_span: tuple[float, float]
    y0: np.ndarray
    labels: list[str]
    rhs: ArrayRhs
    reference_method: str = "DOP853"
    baseline_rhs: ArrayRhs | None = None

    @property
    def candidate_rhs(self):
        return self.baseline_rhs or self.rhs


def lorenz_rhs(t, state):
    del t
    x, y, z = state
    return np.array([
        10.0 * (y - x),
        x * (28.0 - z) - y,
        x * y - (8.0 / 3.0) * z,
    ])


def oscillator_true_rhs(t, state):
    del t
    y, velocity = state
    return np.array([velocity, -0.1 * velocity - 4.0 * y])


def oscillator_wrong_rhs(t, state):
    del t
    y, velocity = state
    return np.array([velocity, -0.5 * velocity - 3.0 * y])


def vanderpol_rhs(t, state):
    del t
    y, velocity = state
    mu = 5.0
    return np.array([velocity, mu * (1.0 - y**2) * velocity - y])


def robertson_rhs(t, state):
    del t
    y1, y2, y3 = state
    return np.array([
        -0.04 * y1 + 1.0e4 * y2 * y3,
        0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
        3.0e7 * y2 * y2,
    ])


SYSTEMS = {
    "lorenz": DynamicalSystem(
        key="lorenz",
        name="Lorenz attractor",
        description="Chaotic nonlinear dynamics with sensitive dependence on initial conditions.",
        stiffness="non-stiff / chaotic",
        t_span=(0.0, 10.0),
        y0=np.array([1.0, 1.0, 1.0], dtype=float),
        labels=["x", "y", "z"],
        rhs=lorenz_rhs,
        reference_method="DOP853",
    ),
    "vanderpol": DynamicalSystem(
        key="vanderpol",
        name="Van der Pol oscillator",
        description="Nonlinear oscillator with relaxation behavior.",
        stiffness="moderately stiff",
        t_span=(0.0, 20.0),
        y0=np.array([2.0, 0.0], dtype=float),
        labels=["position", "velocity"],
        rhs=vanderpol_rhs,
        reference_method="Radau",
    ),
    "robertson": DynamicalSystem(
        key="robertson",
        name="Robertson chemical kinetics",
        description="Classic stiff reaction system with very different time scales.",
        stiffness="highly stiff",
        t_span=(0.0, 40.0),
        y0=np.array([1.0, 0.0, 0.0], dtype=float),
        labels=["y1", "y2", "y3"],
        rhs=robertson_rhs,
        reference_method="Radau",
    ),
    "mismatch": DynamicalSystem(
        key="mismatch",
        name="Misspecified damped oscillator",
        description="True dynamics differ from the physics model, making residual correction useful.",
        stiffness="non-stiff / model mismatch",
        t_span=(0.0, 20.0),
        y0=np.array([2.0, 0.0], dtype=float),
        labels=["position", "velocity"],
        rhs=oscillator_true_rhs,
        baseline_rhs=oscillator_wrong_rhs,
        reference_method="DOP853",
    ),
}


def get_system(key):
    try:
        return SYSTEMS[key]
    except KeyError as exc:
        available = ", ".join(sorted(SYSTEMS))
        raise ValueError(f"Unknown system '{key}'. Available systems: {available}") from exc
