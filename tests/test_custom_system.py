"""Custom equation input must not execute Python expressions."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from adaptive_ode import BenchmarkConfig, benchmark
from adaptive_ode.custom_system import build_custom_system


def build(equations="v\n-c*v-k*x", **overrides):
    options = dict(name="Custom", variables_text="x, v", parameters_text="c=0.1, k=4",
                   equations_text=equations, initial_condition_text="2,0", t_start=0, t_end=4)
    options.update(overrides)
    return build_custom_system(**options)


def test_custom_equations_work_through_public_api():
    system = build()
    assert_allclose(system.rhs(0, [2, 0]), [0, -8])
    result = benchmark(system, config=BenchmarkConfig(num_points=81))
    assert result.selected_method is not None
    assert result.ranking[0]["test_mse"] < 1e-10


def test_fractional_parameters_and_allowed_functions():
    system = build("v\n-sin(x)-a*v+t", parameters_text="a=1/3")
    assert_allclose(system.rhs(1, [np.pi/2, 3]), [3, -1])


@pytest.mark.parametrize("expression", ["().__class__", "x[0]", "__import__('os')", "lambda: 1", "[x for x in (1,)]",
    "open('test')", "unknown+x", "sin(x, v)", "x**x", "x**100", "1/0"])
def test_python_constructs_and_invalid_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        build(f"v\n{expression}")


@pytest.mark.parametrize("changes", [{"variables_text": "x,x"}, {"parameters_text": "x=1"},
    {"parameters_text": "c=0.1,c=0.2"}, {"initial_condition_text": "nan,0"}, {"t_end": np.inf}])
def test_ambiguous_and_nonfinite_inputs_are_rejected(changes):
    with pytest.raises(ValueError):
        build(**changes)
