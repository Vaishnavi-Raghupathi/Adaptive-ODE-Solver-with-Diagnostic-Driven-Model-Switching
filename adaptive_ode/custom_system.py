"""Safe custom dynamical-system parsing."""

import ast
import re

import numpy as np
import sympy as sp

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

def _expression(text, symbols):
    """Convert an arithmetic AST to SymPy without evaluating Python source."""
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Invalid equation expression.") from error
    if sum(1 for _ in ast.walk(tree)) > 150:
        raise ValueError("Expression is too complex.")

    def convert(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not np.isfinite(float(node.value)) or abs(node.value) > 1e12:
                raise ValueError("Numeric constants must be finite and have magnitude <= 1e12.")
            return sp.sympify(node.value)
        if isinstance(node, ast.Name):
            if node.id not in symbols:
                raise ValueError(f"Unknown symbol '{node.id}'.")
            value = symbols[node.id]
            if callable(value):
                raise ValueError(f"Function '{node.id}' must be called.")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if not right.is_number or not right.is_real or abs(float(right)) > 16:
                    raise ValueError("Powers require a real numeric exponent with magnitude <= 16.")
                return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = ALLOWED_FUNCTIONS.get(node.func.id)
            if callable(function) and len(node.args) == 1 and not node.keywords:
                return function(convert(node.args[0]))
        raise ValueError("Only numeric arithmetic, declared symbols and allowed one-argument functions are supported.")

    try:
        expression = convert(tree.body)
        if expression.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise ValueError("Expression contains non-finite values.")
        return expression
    except (TypeError, ZeroDivisionError, OverflowError, RecursionError) as error:
        raise ValueError("Invalid numeric expression.") from error


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
        if name in ALLOWED_FUNCTIONS or name == "t" or name in params:
            raise ValueError(f"Reserved or duplicate parameter name '{name}'.")
        params[name] = float(_expression(raw_value.strip(), ALLOWED_FUNCTIONS))
        if not np.isfinite(params[name]):
            raise ValueError("Parameter values must be finite.")
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

    if len(set(variables)) != len(variables) or any(variable in ALLOWED_FUNCTIONS or variable == "t" for variable in variables):
        raise ValueError("Variable names must be unique and cannot shadow time or allowed functions/constants.")

    equations = [line.strip() for line in equations_text.splitlines() if line.strip()]
    if len(equations) != len(variables):
        raise ValueError("Number of equations must match number of state variables.")

    params = _parse_parameters(parameters_text)
    if set(params) & set(variables):
        raise ValueError("Parameter names cannot overlap state variables.")
    y0 = _parse_float_list(initial_condition_text, len(variables), "Initial condition")
    if not np.all(np.isfinite(y0)):
        raise ValueError("Initial conditions must be finite.")
    t_span = (float(t_start), float(t_end))
    if not np.all(np.isfinite(t_span)) or t_span[1] <= t_span[0]:
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
        expression = _expression(equation, local_dict)
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
