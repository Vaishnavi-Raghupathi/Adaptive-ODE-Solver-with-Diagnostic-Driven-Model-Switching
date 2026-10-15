"""Safe custom dynamical-system parsing."""

import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from adaptive_ode.systems import DynamicalSystem


ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}

SAFE_GLOBALS = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}


def _split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_list(text, expected_count, label):
    values = [float(part) for part in _split_csv(text)]
    if len(values) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} values.")
    return np.array(values, dtype=float)


def _parse_parameters(text):
    params = {}
    if not text.strip():
        return params

    for assignment in _split_csv(text):
        if "=" not in assignment:
            raise ValueError(f"Invalid parameter assignment '{assignment}'. Use name=value.")
        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid parameter name '{name}'.")
        params[name] = float(parse_expr(
            raw_value.strip(),
            local_dict=ALLOWED_FUNCTIONS,
            global_dict=SAFE_GLOBALS,
            transformations=standard_transformations,
            evaluate=True,
        ))
    return params


def build_custom_system(
    name,
    variables_text,
    parameters_text,
    equations_text,
    initial_condition_text,
    t_start,
    t_end,
    reference_method="DOP853",
):
    variables = _split_csv(variables_text)
    if not variables:
        raise ValueError("Enter at least one state variable.")
    for variable in variables:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ValueError(f"Invalid variable name '{variable}'.")

    equations = [line.strip() for line in equations_text.splitlines() if line.strip()]
    if len(equations) != len(variables):
        raise ValueError("Number of equations must match number of state variables.")

    params = _parse_parameters(parameters_text)
    y0 = _parse_float_list(initial_condition_text, len(variables), "Initial condition")
    t_span = (float(t_start), float(t_end))
    if t_span[1] <= t_span[0]:
        raise ValueError("End time must be greater than start time.")

    variable_symbols = {name: sp.Symbol(name) for name in variables}
    parameter_symbols = {name: sp.Symbol(name) for name in params}
    local_dict = {
        **ALLOWED_FUNCTIONS,
        **variable_symbols,
        **parameter_symbols,
        "t": sp.Symbol("t"),
    }
    allowed_symbols = set(variable_symbols.values()) | set(parameter_symbols.values()) | {local_dict["t"]}

    parsed_equations = []
    for equation in equations:
        expression = parse_expr(
            equation,
            local_dict=local_dict,
            global_dict=SAFE_GLOBALS,
            transformations=standard_transformations,
            evaluate=True,
        )
        unknown_symbols = expression.free_symbols - allowed_symbols
        if unknown_symbols:
            names = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
            raise ValueError(f"Unknown symbols in equation '{equation}': {names}")
        parsed_equations.append(expression)

    ordered_symbols = [variable_symbols[name] for name in variables]
    parameter_values = {parameter_symbols[name]: value for name, value in params.items()}
    expressions = [sp.N(expression.subs(parameter_values)) for expression in parsed_equations]
    numeric_rhs = sp.lambdify((local_dict["t"], ordered_symbols), expressions, modules="numpy")

    def rhs(t, state):
        values = numeric_rhs(float(t), np.asarray(state, dtype=float))
        return np.asarray(values, dtype=float).reshape(-1)

    return DynamicalSystem(
        key="custom",
        name=name.strip() or "Custom dynamical system",
        description="Custom equation mode: user-specified governing equations.",
        stiffness="user-defined",
        t_span=t_span,
        y0=y0,
        labels=variables,
        rhs=rhs,
        reference_method=reference_method,
    )
