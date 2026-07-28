import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from adaptive_ode.custom_system import build_custom_system
from adaptive_ode.pipeline.run_pipeline import run_custom_pipeline, run_data_pipeline, run_pipeline
from adaptive_ode.systems import SYSTEMS


DIAGNOSTIC_MEANINGS = {
    "heteroscedasticity": "Residual variance changes with prediction level.",
    "autocorrelation": "Residuals remain temporally correlated.",
    "non_stationary": "Residual statistics drift over time.",
    "state_dependence": "Error magnitude depends on predicted state.",
}


EDGE_CASES_SUPPORTED = [
    "Invalid custom equation syntax, unknown variables, mismatched equation counts, and invalid initial-condition length.",
    "Non-increasing or duplicated uploaded time values.",
    "Non-finite uploaded values such as NaN or infinity.",
    "Solver failures, unstable trajectories, and non-finite predictions are marked as failed instead of ranked.",
]


EDGE_CASES_LIMITED = [
    "Delay differential equations, PDEs, DAEs, event-triggered systems, discontinuous RHS functions, and piecewise dynamics.",
    "Very high-dimensional systems or extremely long time spans that exceed Streamlit/cloud runtime limits.",
    "Equations requiring functions outside the allowed parser list: sin, cos, tan, exp, log, sqrt, abs, pi, E.",
]


def _format_candidate_table(rows):
    formatted = []
    for row in rows:
        formatted.append({
            "Recommended": "Yes" if row["Recommended"] else "",
            "Solver": row["Solver"],
            "Family": row["Family"],
            "Status": row["Status"],
            "MSE": f"{row['MSE']:.6g}" if row["MSE"] is not None else "N/A",
            "RMSE": f"{row['RMSE']:.6g}" if row["RMSE"] is not None else "N/A",
            "MAE": f"{row['MAE']:.6g}" if row["MAE"] is not None else "N/A",
            "Runtime (s)": f"{row['Runtime (s)']:.4f}",
        })
    return formatted


def _candidate_metrics_df(results):
    return pd.DataFrame(_format_candidate_table(results["candidate_results"]))


def _numeric_candidate_df(results):
    rows = [row for row in results["candidate_results"] if row["MSE"] is not None]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("MSE", ascending=True).reset_index(drop=True)
    df["Rank"] = df.index + 1
    best_mse = float(df.loc[0, "MSE"])
    df["Error vs best"] = df["MSE"] / best_mse if best_mse > 0 else 1.0
    df["Selection"] = np.where(df["Recommended"], "Selected", "Candidate")
    return df


def _format_diagnostics_table(results):
    tests = (results.get("diagnostics") or {}).get("test_results", {})
    rows = []
    for name, meaning in DIAGNOSTIC_MEANINGS.items():
        flagged = bool(tests.get(name, False))
        rows.append({
            "Diagnostic": name,
            "Result": "Flagged" if flagged else "Pass",
            "Meaning": meaning,
        })
    return rows


def _available_predictions(results):
    predictions = {}
    for item in results.get("candidate_predictions", []):
        if item["status"] == "ok" and item["prediction"] is not None:
            predictions[item["name"]] = np.asarray(item["prediction"])
    return predictions


def _trajectory_download_df(results, selected_candidates):
    t = np.asarray(results["t"])
    labels = results["system"]["labels"]
    data = {"t": t}
    y_reference = np.asarray(results["y_reference"])
    y_observed = np.asarray(results["y_observed"])

    for dim, label in enumerate(labels):
        data[f"reference_{label}"] = y_reference[:, dim]
        data[f"observed_{label}"] = y_observed[:, dim]

    predictions = _available_predictions(results)
    for solver in selected_candidates:
        if solver not in predictions:
            continue
        safe_solver = solver.lower().replace(" ", "_").replace("+", "plus")
        for dim, label in enumerate(labels):
            data[f"{safe_solver}_{label}"] = predictions[solver][:, dim]
    return pd.DataFrame(data)


def _trajectory_figure(results, selected_dims, selected_candidates, show_reference, show_observed, template):
    t = np.asarray(results["t"])
    labels = results["system"]["labels"]
    y_reference = np.asarray(results["y_reference"])
    y_observed = np.asarray(results["y_observed"])
    predictions = _available_predictions(results)
    fig = go.Figure()

    for dim in selected_dims:
        label = labels[dim]
        if show_reference:
            fig.add_trace(go.Scatter(
                x=t,
                y=y_reference[:, dim],
                mode="lines",
                name=f"reference: {label}",
                line={"width": 3, "color": "#111827"},
            ))
        if show_observed:
            fig.add_trace(go.Scatter(
                x=t,
                y=y_observed[:, dim],
                mode="markers",
                name=f"observed: {label}",
                marker={"size": 5, "opacity": 0.45},
            ))
        for solver in selected_candidates:
            if solver not in predictions:
                continue
            fig.add_trace(go.Scatter(
                x=t,
                y=predictions[solver][:, dim],
                mode="lines",
                name=f"{solver}: {label}",
                line={"width": 2, "dash": "dash"},
            ))

    fig.update_layout(
        title="Trajectory comparison",
        xaxis_title="Time",
        yaxis_title="State value",
        hovermode="x unified",
        template=template,
        legend_title="Series",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return fig


def _residual_figure(results, selected_dims, selected_solver, template):
    t = np.asarray(results["t"])
    labels = results["system"]["labels"]
    y_reference = np.asarray(results["y_reference"])
    predictions = _available_predictions(results)
    fig = go.Figure()

    if selected_solver not in predictions:
        return fig

    residuals = y_reference - predictions[selected_solver]
    for dim in selected_dims:
        fig.add_trace(go.Scatter(
            x=t,
            y=residuals[:, dim],
            mode="lines",
            name=f"{selected_solver}: {labels[dim]}",
        ))

    fig.add_hline(y=0.0, line_dash="dash", line_color="#111827", opacity=0.5)
    fig.update_layout(
        title="Residuals against reference / observed target",
        xaxis_title="Time",
        yaxis_title="Residual",
        hovermode="x unified",
        template=template,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return fig


def _ranking_figure(results, log_scale, template):
    df = _numeric_candidate_df(results)
    if df.empty:
        return go.Figure()

    fig = px.bar(
        df,
        x="MSE",
        y="Solver",
        color="Selection",
        orientation="h",
        hover_data=["RMSE", "MAE", "Runtime (s)", "Family"],
        template=template,
        color_discrete_map={"Selected": "#16a34a", "Candidate": "#64748b"},
        title="Solver ranking by held-out MSE",
    )
    if log_scale:
        fig.update_xaxes(type="log")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Held-out MSE",
        yaxis_title="Solver",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return fig


def _relative_race_figure(results, template):
    df = _numeric_candidate_df(results)
    if df.empty:
        return go.Figure()

    fig = px.bar(
        df,
        x="Error vs best",
        y="Solver",
        color="Selection",
        orientation="h",
        text=df["Error vs best"].map(lambda value: f"{value:.2f}x"),
        hover_data=["MSE", "RMSE", "MAE", "Runtime (s)", "Family"],
        color_discrete_map={"Selected": "#16a34a", "Candidate": "#f97316"},
        template=template,
        title="Solver race: relative error compared with the winner",
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#111827", opacity=0.6)
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Relative held-out MSE, lower is better",
        yaxis_title="Solver",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return fig


def _winner_panel(results):
    df = _numeric_candidate_df(results)
    if df.empty:
        return

    best = df.iloc[0]
    second = df.iloc[1] if len(df) > 1 else None
    margin_text = "Only successful candidate"
    if second is not None:
        ratio = float(second["MSE"] / best["MSE"]) if best["MSE"] > 0 else float("inf")
        margin_text = f"{ratio:.2f}x lower MSE than next best" if ratio >= 1 else "Closest solver by MSE"

    st.markdown(
        f"""
        <div style="
            border: 1px solid #bbf7d0;
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%);
            border-radius: 14px;
            padding: 18px 20px;
            margin: 10px 0 18px 0;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        ">
            <div style="font-size: 13px; color: #15803d; font-weight: 700; letter-spacing: 0.04em;">
                SELECTED SOLVER
            </div>
            <div style="font-size: 28px; font-weight: 800; color: #0f172a; margin-top: 4px;">
                {best["Solver"]}
            </div>
            <div style="color: #334155; margin-top: 6px;">
                {best["Family"]} · {margin_text}
            </div>
            <div style="display: flex; gap: 18px; flex-wrap: wrap; margin-top: 14px;">
                <div><b>MSE</b><br>{best["MSE"]:.6g}</div>
                <div><b>RMSE</b><br>{best["RMSE"]:.6g}</div>
                <div><b>MAE</b><br>{best["MAE"]:.6g}</div>
                <div><b>Runtime</b><br>{best["Runtime (s)"]:.4f}s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _download_buttons(metrics_df, trajectory_df, trajectory_fig, residual_fig, ranking_fig, race_fig):
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download metrics CSV",
        metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="solver_metrics.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Download trajectory CSV",
        trajectory_df.to_csv(index=False).encode("utf-8"),
        file_name="solver_trajectories.csv",
        mime="text/csv",
    )
    html = "\n".join([
        trajectory_fig.to_html(full_html=False, include_plotlyjs="cdn"),
        residual_fig.to_html(full_html=False, include_plotlyjs=False),
        ranking_fig.to_html(full_html=False, include_plotlyjs=False),
        race_fig.to_html(full_html=False, include_plotlyjs=False),
    ])
    col3.download_button(
        "Download plots HTML",
        html.encode("utf-8"),
        file_name="solver_plots.html",
        mime="text/html",
    )


def _custom_system_inputs():
    st.sidebar.header("Custom Equation")
    with st.sidebar.expander("Equation settings", expanded=True):
        name = st.text_input("System name", "Custom Lorenz system")
        variables = st.text_input("State variables", "x, y, z")
        parameters = st.text_input("Parameters", "sigma=10, rho=28, beta=8/3")
        equations = st.text_area(
            "Equations, one derivative per line",
            "sigma*(y - x)\nx*(rho - z) - y\nx*y - beta*z",
            height=130,
        )
        initial_condition = st.text_input("Initial condition", "1, 1, 1")
        t_start = st.number_input("Start time", value=0.0)
        t_end = st.number_input("End time", value=10.0)
        reference_method = st.selectbox("Reference method", ["DOP853", "Radau", "BDF", "LSODA"], index=0)
    return {
        "name": name,
        "variables": variables,
        "parameters": parameters,
        "equations": equations,
        "initial_condition": initial_condition,
        "t_start": t_start,
        "t_end": t_end,
        "reference_method": reference_method,
    }


def _edge_case_panel():
    with st.expander("Edge Case Handling", expanded=False):
        st.markdown("**Handled gracefully**")
        for item in EDGE_CASES_SUPPORTED:
            st.markdown(f"- {item}")
        st.markdown("**Current limitations**")
        for item in EDGE_CASES_LIMITED:
            st.markdown(f"- {item}")


def _show_results(results):
    recommended = results["recommended_solver"]
    system = results["system"]

    st.subheader(system["name"])
    st.write(system["description"])

    if recommended is None:
        st.error("No solver completed successfully.")
        return

    _winner_panel(results)
    st.write(results["recommendation_reason"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Held-out MSE", f"{recommended['MSE']:.6g}")
    col2.metric("Held-out RMSE", f"{recommended['RMSE']:.6g}")
    col3.metric("Runtime", f"{recommended['Runtime (s)']:.4f}s")

    metrics_df = _candidate_metrics_df(results)
    st.subheader("Candidate Ranking")
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    labels = results["system"]["labels"]
    prediction_names = list(_available_predictions(results))
    default_candidates = [recommended["Solver"]] if recommended["Solver"] in prediction_names else prediction_names[:1]

    with st.expander("Graph Controls", expanded=True):
        control_col1, control_col2 = st.columns(2)
        selected_labels = control_col1.multiselect("State dimensions", labels, default=labels[: min(3, len(labels))])
        selected_candidates = control_col1.multiselect("Solver curves", prediction_names, default=default_candidates)
        residual_solver = control_col2.selectbox(
            "Residual solver",
            prediction_names,
            index=prediction_names.index(default_candidates[0]) if default_candidates else 0,
        )
        template = control_col2.selectbox("Plot style", ["plotly_white", "plotly", "ggplot2", "simple_white"], index=0)
        show_reference = control_col2.checkbox("Show reference/target", value=True)
        show_observed = control_col2.checkbox("Show observed points", value=results["system"]["key"] in {"uploaded"})
        log_ranking = control_col2.checkbox("Log-scale ranking", value=True)

    selected_dims = [labels.index(label) for label in selected_labels] or [0]
    trajectory_fig = _trajectory_figure(
        results,
        selected_dims,
        selected_candidates,
        show_reference,
        show_observed,
        template,
    )
    residual_fig = _residual_figure(results, selected_dims, residual_solver, template)
    ranking_fig = _ranking_figure(results, log_ranking, template)
    race_fig = _relative_race_figure(results, template)
    trajectory_df = _trajectory_download_df(results, selected_candidates)

    st.subheader("Interactive Plots")
    st.plotly_chart(trajectory_fig, width="stretch")
    st.plotly_chart(residual_fig, width="stretch")
    chart_tab1, chart_tab2 = st.tabs(["Relative Solver Race", "Raw MSE Ranking"])
    with chart_tab1:
        st.plotly_chart(race_fig, width="stretch")
    with chart_tab2:
        st.plotly_chart(ranking_fig, width="stretch")
    _download_buttons(metrics_df, trajectory_df, trajectory_fig, residual_fig, ranking_fig, race_fig)

    with st.expander("Diagnostics on Recommended Solver", expanded=False):
        st.table(_format_diagnostics_table(results))

    with st.expander("Candidate Notes", expanded=False):
        for row in results["candidate_results"]:
            st.markdown(f"**{row['Solver']}**: {row['Notes']}")

    _edge_case_panel()


def main():
    st.title("Adaptive Solver Selection")
    st.caption(
        "Benchmarks classical, hybrid, physics-informed, and data-driven candidates, "
        "then recommends the solver with the best held-out performance."
    )

    mode = st.sidebar.radio("Input mode", ["Built-in system", "Custom equation", "Upload data"])
    uploaded_file = None
    time_column = None
    state_columns = []
    custom_inputs = None
    system_key = None

    if mode == "Built-in system":
        system_options = {system.name: key for key, system in SYSTEMS.items()}
        st.sidebar.header("System")
        system_name = st.sidebar.selectbox("Dynamical system", list(system_options.keys()))
        system_key = system_options[system_name]
        selected_system = SYSTEMS[system_key]
        st.sidebar.write(selected_system.description)
        st.sidebar.caption(f"Regime: {selected_system.stiffness}")
    elif mode == "Custom equation":
        custom_inputs = _custom_system_inputs()
    else:
        st.sidebar.header("Upload Data")
        uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])
        if uploaded_file is not None:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            numeric_columns = list(preview_df.select_dtypes(include="number").columns)
            if len(numeric_columns) >= 2:
                time_column = st.sidebar.selectbox("Time column", numeric_columns, index=0)
                default_states = [col for col in numeric_columns if col != time_column]
                state_columns = st.sidebar.multiselect(
                    "State columns",
                    numeric_columns,
                    default=default_states,
                )
                with st.expander("Uploaded Data Preview", expanded=False):
                    st.dataframe(preview_df.head(20), width="stretch")
            else:
                st.sidebar.warning("CSV needs at least one time column and one state column.")

    st.sidebar.header("Benchmark")
    num_points = st.sidebar.slider(
        "Trajectory points",
        80,
        800,
        240,
        step=20,
        disabled=mode == "Upload data",
        help="Number of time samples generated for known-equation systems.",
    )
    train_fraction = st.sidebar.slider(
        "Training fraction",
        0.4,
        0.85,
        0.7,
        step=0.05,
        help="Fraction of points used to fit hybrid, PINN, or data-driven candidates; remaining points are held out for scoring.",
    )
    noise_std = st.sidebar.slider(
        "Observation noise",
        0.0,
        1.0,
        0.0,
        step=0.05,
        disabled=mode == "Upload data",
        help="Gaussian noise added to the training observations for known-equation benchmark runs.",
    )
    include_pinn = st.sidebar.checkbox(
        "Include PINN surrogate",
        value=False,
        disabled=mode != "Built-in system",
        help="Runs a lightweight physics-informed neural surrogate. Available for built-in systems.",
    )
    pinn_epochs = st.sidebar.slider("PINN epochs", 50, 600, 200, step=50, disabled=not include_pinn)

    run_clicked = st.sidebar.button("Run Solver Selection")

    if run_clicked:
        with st.spinner("Running solver candidates..."):
            try:
                if mode == "Upload data":
                    if uploaded_file is None:
                        st.error("Upload a CSV file first.")
                        return
                    if time_column is None or not state_columns:
                        st.error("Select one time column and at least one state column.")
                        return
                    data_frame = pd.read_csv(uploaded_file)
                    raw = run_data_pipeline(
                        data_frame[time_column].to_numpy(),
                        data_frame[state_columns].to_numpy(),
                        labels=state_columns,
                        config={
                            "train_fraction": train_fraction,
                            "save_plots": False,
                        },
                    )
                elif mode == "Custom equation":
                    custom_system = build_custom_system(
                        custom_inputs["name"],
                        custom_inputs["variables"],
                        custom_inputs["parameters"],
                        custom_inputs["equations"],
                        custom_inputs["initial_condition"],
                        custom_inputs["t_start"],
                        custom_inputs["t_end"],
                        custom_inputs["reference_method"],
                    )
                    raw = run_custom_pipeline(custom_system, {
                        "noise_std": noise_std,
                        "num_points": num_points,
                        "train_fraction": train_fraction,
                        "include_pinn": False,
                        "save_plots": False,
                    })
                else:
                    raw = run_pipeline({
                        "system_key": system_key,
                        "noise_std": noise_std,
                        "num_points": num_points,
                        "train_fraction": train_fraction,
                        "include_pinn": include_pinn,
                        "pinn_epochs": pinn_epochs,
                        "save_plots": False,
                    })
                st.session_state["results"] = raw
            except Exception as exc:
                st.error(f"Solver selection failed: {exc}")
                return

    if "results" not in st.session_state:
        st.info("Choose an input mode and click Run Solver Selection.")
        return

    _show_results(st.session_state["results"])


main()
