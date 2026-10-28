"""Interactive client of the public API; also bundled in the installed wheel."""

from dataclasses import asdict
import json
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adaptive_ode import BenchmarkConfig, benchmark, benchmark_data
from adaptive_ode.systems import SYSTEMS


def main():
    st.set_page_config(page_title="Adaptive Solver Bench", layout="wide")
    st.title("Adaptive Solver Bench")
    st.write("Find which tested solver or forecasting model works best for your time-series dataset.")
    st.caption("Choose data and optional equations, compare methods, and export the evidence behind the recommendation.")
    mode = st.sidebar.radio("Input", ["Uploaded CSV", "Independent trajectories", "Built-in system", "Custom equations"])
    if mode == "Independent trajectories":
        from adaptive_ode.trajectory_demo import render
        render()
        return
    system = None
    frame = None
    time_column = None
    states = None
    custom = None
    equation_spec = None
    source_id = None
    if mode == "Built-in system":
        keys = ["mismatch", "lorenz", "vanderpol", "robertson"]
        system = st.sidebar.selectbox("System", keys, format_func=lambda key: SYSTEMS[key].name)
        st.caption(SYSTEMS[system].description)
    elif mode == "Uploaded CSV":
        example = st.sidebar.checkbox("Use example oscillator dataset", value=True)
        uploaded = None if example else st.sidebar.file_uploader("Numeric time-series CSV", type=["csv"])
        if example or uploaded is not None:
            try:
                if example:
                    from adaptive_ode.oscillator import OscillatorParameters, analytical_trajectory
                    times = np.linspace(0, 10, 201)
                    values = analytical_trajectory(times, [2, 0], OscillatorParameters(.1, 4))
                    frame = pd.DataFrame({"time": times, "position": values[:, 0], "velocity": values[:, 1]})
                    source_id = "example-oscillator-v1"
                    st.info("Example: a synthetic, fully observed damped oscillator. Replace it with your own CSV to evaluate your data.")
                    st.download_button("Download example CSV", frame.to_csv(index=False), "oscillator.csv", "text/csv")
                else:
                    frame = pd.read_csv(uploaded)
                    source_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
                numeric = list(frame.select_dtypes(include="number").columns)
                if len(numeric) < 2:
                    st.error("Provide one numeric time column and at least one numeric state column.")
                    return
                time_column = st.sidebar.selectbox("Time column", numeric, index=numeric.index("time") if "time" in numeric else 0)
                states = st.sidebar.multiselect("State columns (equation order)", [column for column in numeric if column != time_column],
                                                default=[column for column in numeric if column != time_column])
                st.dataframe(frame.head(10), hide_index=True)
                if states and st.sidebar.checkbox("I have governing equations", value=False):
                    variables = [f"y{index}" for index in range(len(states))]
                    st.write("Equation variable mapping: " + ", ".join(f"{var} = {column}" for var, column in zip(variables, states)))
                    derivatives = ["y1", "-c*y1-k*y0"] if len(states) == 2 else [f"-{var}" for var in variables]
                    spec = {"variables": variables, "parameters": {"c": .1, "k": 4} if len(states) == 2 else {},
                            "derivatives": derivatives}
                    equation_spec = st.text_area("Equation specification (JSON)", json.dumps(spec, indent=2), height=220)
                    st.caption("Replace these example equations with your model. Units and variable order must match the CSV. Parameters are supplied, not automatically calibrated.")
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
                return
        st.caption("Use one numeric trajectory with increasing, unique times and finite states. Do not select categories or external inputs as states, or concatenate separate experiments into one physical trajectory.")
    else:
        custom = {
            "name": st.sidebar.text_input("Name", "Custom oscillator"),
            "variables": st.sidebar.text_input("Variables", "x, v"),
            "parameters": st.sidebar.text_input("Parameters", "c=0.1, k=4"),
            "equations": st.sidebar.text_area("One derivative per line", "v\n-c*v-k*x"),
            "initial": st.sidebar.text_input("Initial state", "2, 0"),
            "end": st.sidebar.number_input("End time", min_value=0.1, value=10.0),
        }
        st.caption("Compare integration methods against a tight reference generated from these equations.")
    points = st.sidebar.slider("Trajectory points", 80, 800, 201, disabled=mode == "Uploaded CSV")
    train = st.sidebar.slider("Training fraction", .3, .7, .6, .05)
    validation = st.sidebar.slider("Validation fraction", .1, .25, .2, .05)
    metric = st.sidebar.selectbox("Selection metric", ["normalized_rmse", "mse"],
                                 help="Normalized RMSE uses training-only state scales; MSE keeps original units.")
    include_ml = st.sidebar.checkbox("Include neural forecasting candidates", value=False)
    policy_label = st.sidebar.selectbox("Recommendation rule", ["Lowest validation error", "Fastest meeting accuracy target"])
    policy = "min_error" if policy_label.startswith("Lowest") else "fastest_within_target"
    accuracy_target = st.sidebar.number_input("Maximum validation score", min_value=0., value=.05, step=.01, format="%.4g") if policy == "fastest_within_target" else None
    selection_cost = st.sidebar.selectbox("Runtime objective", ["prediction_seconds", "fit_plus_prediction_seconds"])
    repeats = st.sidebar.select_slider("Prediction timing repeats", options=[1, 3, 5], value=3)
    sweep = st.sidebar.checkbox("Sweep solver tolerances and RK4 steps", value=False)
    st.sidebar.caption(f"Final future test fraction: {1-train-validation:.0%}. Final test errors never select the winner.")
    signature = json.dumps([mode, system, source_id, time_column, states, custom, equation_spec,
                            points, train, validation, metric, include_ml, policy, accuracy_target, selection_cost, repeats, sweep], sort_keys=True)
    if st.sidebar.button("Run benchmark", type="primary"):
        st.session_state.pop("benchmark_result", None)
        try:
            config = BenchmarkConfig(num_points=points, train_fraction=train, validation_fraction=validation,
                                     include_ml=include_ml, selection_metric=metric, selection_policy=policy,
                                     accuracy_target=accuracy_target, selection_cost=selection_cost,
                                     timing_repeats=repeats, solver_sweep=sweep)
            with st.spinner("Fitting and evaluating candidates…"):
                if mode == "Uploaded CSV":
                    if frame is None or not states:
                        st.error("Upload a CSV or choose the example, then select state columns.")
                        return
                    times, values = frame[time_column].to_numpy(), frame[states].to_numpy()
                    rhs = None
                    if equation_spec is not None:
                        from adaptive_ode.custom_system import build_observed_system
                        rhs = build_observed_system(json.loads(equation_spec), times, values).rhs
                    result = benchmark_data(times, values, config=config, rhs=rhs)
                    result.context.update({"time_column": time_column, "state_columns": states, "source_id": source_id,
                                           "equation_spec": json.loads(equation_spec) if equation_spec is not None else None})
                elif mode == "Custom equations":
                    from adaptive_ode.custom_system import build_custom_system
                    supplied = build_custom_system(custom["name"], custom["variables"], custom["parameters"],
                                                   custom["equations"], custom["initial"], 0, custom["end"])
                    result = benchmark(supplied, config=config)
                    result.context["equation_spec"] = custom
                else:
                    result = benchmark(system, config=config)
            st.session_state["benchmark_result"] = result
            st.session_state["benchmark_signature"] = signature
        except Exception as error:
            st.error(f"Benchmark failed: {error}")
            return
    result = st.session_state.get("benchmark_result")
    if result is None or st.session_state.get("benchmark_signature") != signature:
        st.info("Choose a dataset and run the benchmark. Changing data or benchmark settings requires a new run.")
        st.write("Without equations: compare simple forecasts and learned dynamics. With equations: also compare RK4, RK45, DOP853, BDF, Radau and LSODA.")
        return
    assessment = result.assessment
    st.subheader(result.name)
    st.write(assessment["comparison_scope"])
    st.write(assessment["selection_rule"])
    if result.selected_method is None:
        st.warning("No candidate meets the selection rule. Inspect validation scores and failures below; no fallback winner was chosen.")
    else:
        selected = next(row for row in result.rows if row["selected"])
        a, b, c = st.columns(3)
        a.metric("Selected on validation", result.selected_method)
        b.metric(f"Validation {metric}", f"{selected['validation_score']:.3e}")
        c.metric("Final test normalized RMSE", f"{selected['test_normalized_rmse']:.3e}"
                 if selected['test_normalized_rmse'] is not None else "Unavailable")
    for warning in assessment["warnings"]:
        st.warning(warning)
    with st.expander("Dataset profile and recommendation scope"):
        st.json(assessment["profile"])
        for limitation in assessment["limitations"]:
            st.write(limitation)
    table = pd.DataFrame(result.ranking)
    columns = ["method", "variant", "family", "selected", "meets_accuracy_target", "status", "validation_score", "validation_mse",
               "validation_normalized_rmse", "test_rmse", "test_normalized_rmse", "prediction_seconds", "search_seconds"]
    st.dataframe(table[columns], hide_index=True, width="stretch")
    st.caption("Rows are ordered by the chosen validation metric. Search time includes all tried variants; failures remain visible.")
    st.write("Accuracy/runtime trade-off methods: " + (", ".join(assessment["accuracy_runtime_pareto_methods"]) or "none"))
    st.caption("The trade-off set uses validation error and the chosen runtime objective. Timing is local; final test errors do not select the winner.")
    labels = states if mode == "Uploaded CSV" else list(range(result.target.shape[1]))
    state = st.selectbox("State dimension", list(range(result.target.shape[1])), format_func=lambda i: str(labels[i]))
    successful = [row["method"] for row in result.ranking if row["status"] == "ok"]
    default_plot = list(dict.fromkeys(successful[:3] + ([result.selected_method] if result.selected_method else [])))
    plotted = st.multiselect("Methods to plot", successful, default=default_plot,
                            help="Start with the top three validation methods. Add other methods to inspect their predictions.")
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=result.t, y=result.target[:, state], name=result.reference_kind,
                                line={"color": "black", "width": 3}))
    for row in result.ranking:
        if row["status"] == "ok" and row["method"] in plotted:
            figure.add_trace(go.Scatter(x=result.t, y=result.predictions[row["prediction_key"]][:, state], name=row["method"]))
    figure.add_vline(x=float(result.t[result.split["validation"][0]]), line_dash="dot")
    figure.add_vrect(x0=float(result.t[result.split["test"][0]]), x1=float(result.t[-1]), fillcolor="gray", opacity=.12, line_width=0)
    figure.update_layout(xaxis_title="Time", yaxis_title=str(labels[state]), hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    st.caption("Dotted line: validation starts. Shaded region: final future test segment. Predictions start at the original initial state, without true-state resets at split boundaries.")
    st.download_button("Download recommendation report", result.report(), "recommendation.md", "text/markdown")
    csv_data = table.drop(columns=["details", "warnings", "residual_summary"]).to_csv(index=False)
    st.download_button("Download metrics CSV", csv_data, "benchmark_metrics.csv", "text/csv")
    record = {"schema_version": 2, "config": asdict(result.config), "selected_method": result.selected_method,
              "assessment": assessment, "context": result.context,
              "rows": result.rows, "trials": result.trials,
              "split_indices": {key: values.tolist() for key, values in result.split.items()}}
    st.download_button("Download evaluation JSON", json.dumps(record, indent=2, allow_nan=False), "benchmark_results.json", "application/json")
    predictions = pd.DataFrame({"time": result.t})
    for index, label in enumerate(labels):
        predictions[f"target:{label}"] = result.target[:, index]
    for row in result.ranking:
        if row["status"] == "ok":
            for index, label in enumerate(labels):
                predictions[f"{row['method']}:{label}"] = result.predictions[row["prediction_key"]][:, index]
    st.download_button("Download trajectories CSV", predictions.to_csv(index=False), "benchmark_predictions.csv", "text/csv")
    with st.expander("Test residual summaries, warnings, and failed attempts"):
        for row in result.rows:
            st.write(row["method"], row["residual_summary"])
        for trial in result.trials:
            if trial["message"] or trial["warnings"]:
                st.write(trial["variant"], trial["message"], trial["warnings"])
    st.caption("A recommendation applies to the tested methods and forecast horizon. A CSV alone cannot establish stiffness, chaos or correct governing equations.")


if __name__ == "__main__":
    main()
