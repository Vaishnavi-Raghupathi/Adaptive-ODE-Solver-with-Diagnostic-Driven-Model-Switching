"""Interactive client of the public API; also bundled in the installed wheel."""

from dataclasses import asdict
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adaptive_ode import BenchmarkConfig, benchmark, benchmark_data
from adaptive_ode.systems import SYSTEMS


def main():
    st.set_page_config(page_title="Adaptive Solver Bench", layout="wide")
    st.title("Adaptive Solver Bench")
    st.write("Compare numerical solvers and learned models. Fit on training data, select on validation data, and report future test error separately.")
    mode = st.sidebar.radio("Input", ["Built-in system", "Uploaded CSV", "Custom equations"])
    system = None
    frame = None
    time_column = None
    states = None
    custom = None
    if mode == "Built-in system":
        keys = ["mismatch", "lorenz", "vanderpol", "robertson"]
        system = st.sidebar.selectbox("System", keys, format_func=lambda key: SYSTEMS[key].name)
        st.caption(SYSTEMS[system].description)
    elif mode == "Uploaded CSV":
        uploaded = st.sidebar.file_uploader("Numeric time-series CSV", type=["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                numeric = list(frame.select_dtypes(include="number").columns)
                if len(numeric) < 2:
                    st.error("Provide one numeric time column and at least one numeric state column.")
                    return
                time_column = st.sidebar.selectbox("Time column", numeric)
                states = st.sidebar.multiselect("State columns", [column for column in numeric if column != time_column],
                                                default=[column for column in numeric if column != time_column])
                st.dataframe(frame.head(10), hide_index=True)
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
                return
        st.caption("Uploaded observations are evaluation targets and may contain measurement noise. Time must be unique and increasing.")
    else:
        custom = {
            "name": st.sidebar.text_input("Name", "Custom oscillator"),
            "variables": st.sidebar.text_input("Variables", "x, v"),
            "parameters": st.sidebar.text_input("Parameters", "c=0.1, k=4"),
            "equations": st.sidebar.text_area("One derivative per line", "v\n-c*v-k*x"),
            "initial": st.sidebar.text_input("Initial state", "2, 0"),
            "end": st.sidebar.number_input("End time", min_value=0.1, value=10.0),
        }
        st.caption("Custom equations use the existing restricted SymPy expression parser. PINN candidates are not part of this validated API.")
    points = st.sidebar.slider("Trajectory points", 80, 800, 201, disabled=mode == "Uploaded CSV")
    train = st.sidebar.slider("Training fraction", .3, .7, .6, .05)
    validation = st.sidebar.slider("Validation fraction", .1, .25, .2, .05)
    include_ml = st.sidebar.checkbox("Include ML candidates", value=False)
    st.sidebar.caption(f"Final future test fraction: {1-train-validation:.0%}. Recommendations use validation MSE, not final test error or runtime.")
    if st.sidebar.button("Run benchmark", type="primary"):
        try:
            config = BenchmarkConfig(num_points=points, train_fraction=train, validation_fraction=validation, include_ml=include_ml)
            with st.spinner("Fitting and evaluating candidates…"):
                if mode == "Uploaded CSV":
                    if frame is None or not states:
                        st.error("Upload a CSV and select state columns.")
                        return
                    result = benchmark_data(frame[time_column].to_numpy(), frame[states].to_numpy(), config=config)
                elif mode == "Custom equations":
                    from adaptive_ode.custom_system import build_custom_system
                    supplied = build_custom_system(custom["name"], custom["variables"], custom["parameters"],
                                                   custom["equations"], custom["initial"], 0, custom["end"])
                    result = benchmark(supplied, config=config)
                else:
                    result = benchmark(system, config=config)
            st.session_state["benchmark_result"] = result
        except Exception as error:
            st.error(f"Benchmark failed: {error}")
            return
    result = st.session_state.get("benchmark_result")
    if result is None:
        st.info("Choose equations or observations and run a benchmark.")
        return
    st.subheader(result.name)
    if result.selected_method is None:
        st.error("No candidate returned a valid complete trajectory. Inspect the recorded failures below.")
    else:
        selected = next(row for row in result.rows if row["selected"])
        a, b, c = st.columns(3)
        a.metric("Selected on validation", result.selected_method)
        b.metric("Validation MSE", f"{selected['validation_mse']:.3e}")
        c.metric("Final test MSE", f"{selected['test_mse']:.3e}" if selected['test_mse'] is not None else "Unavailable")
    table = pd.DataFrame(result.ranking)
    columns = ["method", "selected", "status", "validation_mse", "test_mse", "test_rmse", "search_seconds"]
    st.dataframe(table[columns], hide_index=True, width="stretch")
    st.caption("Rows are ordered by validation error. Search time includes all tried variants; candidate failures remain visible. Lower test error does not change the selected method.")
    state = st.selectbox("State dimension", list(range(result.target.shape[1])))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=result.t, y=result.target[:, state], name=result.reference_kind,
                                line={"color": "black", "width": 3}))
    for row in result.ranking:
        if row["status"] == "ok":
            figure.add_trace(go.Scatter(x=result.t, y=result.predictions[row["prediction_key"]][:, state], name=row["method"]))
    figure.add_vline(x=float(result.t[result.split["validation"][0]]), line_dash="dot")
    figure.add_vrect(x0=float(result.t[result.split["test"][0]]), x1=float(result.t[-1]), fillcolor="gray", opacity=.12, line_width=0)
    figure.update_layout(xaxis_title="Time", yaxis_title=f"State {state}", hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    st.caption("Dotted line: validation starts. Shaded region: untouched future test segment.")
    csv_data = table.drop(columns=["details", "warnings", "residual_summary"]).to_csv(index=False)
    st.download_button("Download metrics CSV", csv_data, "benchmark_metrics.csv", "text/csv")
    record = {"config": asdict(result.config), "selected_method": result.selected_method, "rows": result.rows, "trials": result.trials}
    st.download_button("Download evaluation JSON", json.dumps(record, indent=2, allow_nan=False), "benchmark_results.json", "application/json")
    with st.expander("Test residual summaries, warnings, and failed attempts"):
        for row in result.rows:
            st.write(row["method"], row["residual_summary"])
        for trial in result.trials:
            if trial["message"] or trial["warnings"]:
                st.write(trial["variant"], trial["message"], trial["warnings"])
    st.caption("Evidence so far is from synthetic benchmark systems. A low validation score does not guarantee performance on other trajectories or initial states.")


if __name__ == "__main__":
    main()
