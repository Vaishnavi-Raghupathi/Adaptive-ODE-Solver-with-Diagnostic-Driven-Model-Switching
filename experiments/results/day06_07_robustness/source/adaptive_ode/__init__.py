"""Validation-based benchmarking of dynamical-system models and solvers."""

__version__ = "0.1.0"

# Lazy exports keep legacy research imports and optional dependencies isolated.
__all__ = ["benchmark", "benchmark_data", "BenchmarkConfig", "BenchmarkResult", "Candidate", "FittedModel", "evaluate", "DynamicalSystem"]


def __getattr__(name):
    if name in {"benchmark", "benchmark_data"}:
        from adaptive_ode import api
        return getattr(api, name)
    if name in {"BenchmarkConfig", "BenchmarkResult", "Candidate", "FittedModel", "evaluate"}:
        from adaptive_ode import core
        return getattr(core, name)
    if name == "DynamicalSystem":
        from adaptive_ode.systems import DynamicalSystem
        return DynamicalSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
